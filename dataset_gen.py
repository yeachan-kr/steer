import jsonlines
from torch.utils.data import Dataset
from preprocess import preprocess
from transformers import AutoTokenizer
import torch 
import pandas as pd
import json
from datasets import load_dataset
from utils import duplicate

IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

instruction = "{question}"
aqua_instruction = "{question}\n{opt1}\n{opt2}\n{opt3}\n{opt4}\n{opt5}"
class Gen_Dataset(Dataset):
    def __init__(self, task, q_path, l_path, tokenizer, is_train=True, is_ours=False, cq_path=None, ratio_value=0.9, back_ratio_value=0.9): 
        self.q_path = q_path
        self.l_path = l_path
        self.data = []
        self.tokenizer = tokenizer
        self.is_ours = is_ours
        self.task = task
        self.is_train = is_train
        ratio_ = ratio_value # 0 ~ 1 setting
        ## 메모: wngr 0.5 로 훈련하고, 1로 테스트하기.
        back_ratio_ = back_ratio_value
        
        # load question
        if task == "gsm8k":
            if self.is_train:
                self.data = load_dataset("openai/gsm8k", name = "main", split="train")
                self.answers = self.data["answer"]
            else: # validation
                self.data = load_dataset("openai/gsm8k", name = "main", split="test")
                self.answers = self.data["answer"]
        elif task == "aqua":
            if self.is_train:
                self.data = load_dataset("deepmind/aqua_rat", name = "raw", split="train")
                self.answers = self.data["rationale"]
                print(f'# of data: {len(self.data)}')
            else: # validation
                self.data = load_dataset("deepmind/aqua_rat", name = "raw", split="test")
                self.answers = self.data["correct"]
        
        if self.is_ours:
            with open(cq_path, "r", encoding="utf-8") as f:
                knowledges = json.load(f)
            self.postprocessing(knowledges, ratio_, back_ratio_)
            questions = self.data["question"]
            self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            
        if is_train:
            self.prepare_sft_dataset()
        else:
            self.validation()   
    
    def postprocessing(self, knowledges, ratio, back_ratio=0.9):
        def background_processing(knowledge, back_ratio):
            back_knowledge = ""
            idx = 1
            try:
                for cur_knowledge in knowledge["Background_Knowledge"]:
                    if (cur_knowledge["Confidence"]/100) >= back_ratio:
                        back_knowledge += cur_knowledge[f"Knowledge_{idx}"] + ". "
                    idx += 1
                
                return back_knowledge
            except:
                return "Solve the given question accurately. " # instruction
        
        def background_confidence(knowledge):
            confidence = 1
            try:
                for cur_knowledge in knowledge["Background_Knowledge"]:       
                    confidence = confidence * cur_knowledge["Confidence"] / 100
                return confidence
            except:
                confidence = 0
                return confidence
            
        self.knowledges = []
        self.answer = []
        for knowledge in knowledges:
            cur_answer = knowledge["Final_Answer"]
            cur_final_answer_confidence = knowledge["Final_Confidence"]
            if cur_final_answer_confidence is None:
                cur_final_answer_confidence = 0
            try:
                cur_answer_knowledge_confidence = knowledge["Answer_Knowledge"][0]["Confidence"]
                if cur_answer_knowledge_confidence is None:
                    cur_answer_knowledge_confidence = 0
            except:
                cur_answer_knowledge_confidence = 0
                
            cur_answer_confidence = (cur_final_answer_confidence + cur_answer_knowledge_confidence) / 200
            cur_background_confidence = background_confidence(knowledge)
            overall_confidence = cur_answer_confidence * cur_background_confidence # 모든 step의 confidence 곱
            # overall_confidence = cur_final_answer_confidence/100 # 분석 코드 활성화 끄기.
            if (overall_confidence >= ratio) and (not self.is_train): # inference
                if ((cur_answer == None) or ("none" in cur_answer.lower()) or ("neither" in cur_answer.lower()) or duplicate(cur_answer)):
                # if (cur_answer == None):
                    # print("this instance has null answer") # error
                    back_knowledge = background_processing(knowledge, back_ratio)
                    try:
                        back_answer_knowledge = knowledge["Answer_Knowledge"][0]["Knowledge"] # answer knowledge
                        back_answer_knowledge = f'{back_knowledge}{knowledge["Answer_Knowledge"][0]["Knowledge"]}.'
                    except:
                        back_answer_knowledge = back_knowledge
                        
                    self.knowledges.append(back_answer_knowledge)
                    self.answer.append("Not answer")
                else:
                    # only answer_knowledge만 이용 (분석 코드)
                    # back_knowledge = background_processing(knowledge, back_ratio)
                    # back_answer_knowledge = f'{back_knowledge}{knowledge["Answer_Knowledge"][0]["Knowledge"]}.' # 63.76% (STQA); 훈련한 것과 동일.
                    # # back_answer_knowledge = knowledge["Answer_Knowledge"][0]["Knowledge"] # answer knowledge # 61.14% (STQA)
                    # self.knowledges.append(back_answer_knowledge) 
                    
                    # zero-shot 성능 이용.
                    self.knowledges.append("Direct Answer")
                    
                    self.answer.append(cur_answer)
            elif (overall_confidence >= ratio) and self.is_train: # train
                back_knowledge = background_processing(knowledge, back_ratio)
                try:
                    back_answer_knowledge = f'{back_knowledge}{knowledge["Answer_Knowledge"][0]["Knowledge"]}.'  #stqa: 65.07%, csqa: 69.53%
                    # back_answer_knowledge = f'{knowledge["Answer_Knowledge"][0]["Knowledge"]}.' #stqa: 65.50% (내 아이디어를 고려했을 떄 적합하지 않다.)
                    self.knowledges.append(back_answer_knowledge)
                except:
                    self.knowledges.append(back_knowledge)
            else: # train or inference
                if self.is_train:
                    back_answer_knowledge = background_processing(knowledge, back_ratio) # background knowledge만 참고.
                    
                else: # inference
                    back_knowledge = background_processing(knowledge, back_ratio) # background knowledge만 참고.
                    if cur_answer_knowledge_confidence >= back_ratio:
                        try:
                            back_answer_knowledge = f'{back_knowledge}{knowledge["Answer_Knowledge"][0]["Knowledge"]}.' # in stqa answer-answer -> 65.50, answer-back&answer -> 65.50%
                        except:
                            back_answer_knowledge = back_knowledge
                    else:
                        back_answer_knowledge = back_knowledge
                
                if cur_answer is None:
                    self.answer.append("Not answer") # 오답으로 처리.
                else:
                    self.answer.append(cur_answer)
       
                self.knowledges.append(back_answer_knowledge)
                
    def tokenizing(self, lines):
        encoding = self.tokenizer(
            lines,
            return_tensors = "pt",
            add_special_tokens=True,
            padding=True,
            truncation="longest_first",
            # max_length=512
            max_length=768 # OURS
        )
        
        return encoding
    
    def prepare_sft_dataset(self, verbose=True):
        if self.task == "gsm8k":
            if self.is_ours:
                prompts = [preprocess(instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(instruction.format(question=item["question"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "aqua":
            if self.is_ours:
                prompts = [preprocess(aqua_instruction.format(question=ques, opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for ques, item in zip(self.question, self.data)]
            else:
                prompts = [preprocess(aqua_instruction.format(question=item["question"], opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for item in self.data]
        
        answers = [str(item) for item in self.answers]
        
        texts = [str(prompt) + str(answer) + self.tokenizer.eos_token for prompt, answer in zip(prompts, answers)]
        
        self.encoding = self.tokenizing(texts)
        input_ids = self.encoding["input_ids"]
        self.labels = input_ids.clone()
        

        for idx in range(len(self.labels)):
            self.labels[idx,:len(self.tokenizing(prompts[idx])['input_ids'][0])] = IGNORE_INDEX
        
        torch.set_printoptions(profile="full") # for code verifying
        if verbose:
            print(f'sample example: {texts[0]}')
            # print(self.tokenizing(prompts[0]["input_ids"]))
            # print(self.labels[0])
        return None
    
    def validation(self):
        if self.task == "gsm8k":
            if self.is_ours:
                self.inputs = [preprocess(instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                self.inputs = [preprocess(instruction.format(question=item["question"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "aqua":
            if self.is_ours:
                self.inputs = [preprocess(aqua_instruction.format(question=ques, opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for ques, item in zip(self.question, self.data)]
            else:
                self.inputs = [preprocess(aqua_instruction.format(question=item["question"], opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for item in self.data]
        # labels
        self.labels = self.answers
        
        return None
        
        
    def __getitem__(self, index):
        data = {key: val[index] for key, val in self.encoding.items()}
        # data['labels'] = self.labels['input_ids'][index]
        data['labels'] = self.labels[index]
        return data 

    def __len__(self) -> int:
        return len(self.labels)



if __name__ == "__main__":
    """"
    Instruction model에도 INST라는 토큰은 없다.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        print("add pad token")
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        print("add eos token")
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        print("add bos token")
        None
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        print("add unk token")

    token_list = ['[INST]', "[/INST]"]
    special_tokens_dict['additional_special_tokens'] = token_list
    tokenizer.add_special_tokens(special_tokens_dict)

    # print(tokenizer.convert_tokens_to_ids("[INST]"))
    # print(tokenizer.convert_tokens_to_ids("[/INST]"))
    # print(tokenizer.convert_ids_to_tokens(3297))
    

    # tokenizer.padding_side = "right" # standard
    data = Gen_Dataset("aqua", None, None, tokenizer, True, True, "./dataset/aqua/converted_train.json", 0, 0)
    print(data[0])