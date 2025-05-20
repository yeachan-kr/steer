import jsonlines
from torch.utils.data import Dataset
from preprocess import preprocess
from transformers import AutoTokenizer
import torch 
import pandas as pd
import json
from datasets import load_dataset
from utils import duplicate
# input format
piqa_instruction = '[physicaliqa]:\n<goal>{question}</goal>\n<sol1>{sol1}</sol1>\n<sol2>{sol2}</sol2>'
csqa_instruction = '[commonsenseqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>\n<option5>{sol5}</option5>'
stqa_instruction = '[strategyqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>'
qasc_instruction = '[qasc]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>\n<option5>{sol5}</option5>\n<option6>{sol6}</option6>\n<option7>{sol7}</option7>\n<option8>{sol8}</option8>'
arc_instruction = '[arc]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>'
obqa_instruction = '[obqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>'
siqa_instruction = '[siqa]:\n<context>{context}</context>\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>'
wngr_instruction = '[wngr]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>'
boolq_instruction = '[boolq]:\n<passage>{context}</passage>\n<question>{question}</question>\n<option1>True</option1>\n<option2>False</option2>'

chemprot_instruction = '[chemprot]:\n<question>{question}</question>\n<option1>INHIBITOR</option1>\n<option2>SUBSTRATE</option2>\n<option3>INDIRECT-UPREGULATOR</option3>\n<option4>INDIRECT-DOWNREGULATOR</option4>\n<option5>ACTIVATOR</option5>\n<option6>ANTAGONIST</option6>\n<option7>PRODUCT-OF</option7>\n<option8>AGONIST</option8>\n<option9>DOWNREGULATOR</option9>\n<option10>UPREGULATOR</option10>\n<option11>AGONIST-ACTIVATOR</option11>\n<option12>SUBSTRATE_PRODUCT-OF</option12>\n<option13>AGONIST-INHIBITOR</option13>'
acl_arc_instruction = '[acl_arc]:\n<question>{question}</question>\n<option1>Background</option1>\n<option2>Motivation</option2>\n<option3>CompareOrContrast</option3>\n<option4>Uses</option4>\n<option5>Extends</option5>\n<option6>Future</option6>'
scicite_instruction = '[scicite]:\n<question>{question}</question>\n<option1>background</option1>\n<option2>method</option2>\n<option3>result</option3>'


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

# For cls
class CorrectionJSONDataset(Dataset):
    def __init__(self, task, q_path, l_path, tokenizer, is_train=True, is_ours=False, cq_path=None, ratio_value=0.9, back_ratio_value=0.9, upper_ratio=0): 
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
        self.upper_ratio = upper_ratio
        # load question
        if self.task == "piqa":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, back_ratio_)
                
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                questions = [item['goal'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)

            # load label
            with open(l_path, 'r') as file:
                self.answers = [line.strip() for line in file.readlines()]
        elif self.task == "csqa":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, back_ratio_)
                
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                questions = [item['question']['stem'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
            else:
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
        elif self.task == "stqa":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, back_ratio_)

                with open(self.q_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                questions = [item['question'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
                
                self.answers = [1 if line["answer"] else 0 for line in self.data] # True -> 1, False -> 0
            else:
                with open(self.q_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.answers = [1 if line["answer"] else 0 for line in self.data]
        elif self.task == "qasc":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, back_ratio_)
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                questions = [item['question']['stem'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
                self.answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                                if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                                if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                                if line['answerKey'] == "G" else 7 for line in self.data]
            else:
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
            
                self.answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                                if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                                if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                                if line['answerKey'] == "G" else 7 for line in self.data]
        elif self.task == "arc-e":
            if self.is_train:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Easy", split="train")
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_)
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} Knowledge: {know.strip()}' for ques, know in zip(questions, self.knowledges)]

            else:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Easy", split="test")
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} Knowledge: {know.strip()}' for ques, know in zip(questions, self.knowledges)]       
        elif self.task == "arc-h":
            arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                             '1': 0, '2': 1, '3': 2, '4': 3}
            
            if self.is_train:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="train")
                answers = self.data["answerKey"]
                self.answers = [arc_label_map[str(answer)] for answer in answers]
                print(f'first # of answer: {len(self.answers)}')
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
                answers = self.data["answerKey"]
                self.answers = [arc_label_map[str(answer)] for answer in answers]
                print(f'first # of answer: {len(self.answers)}')
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "obqa":
            if self.is_train:
                self.data = load_dataset("allenai/openbookqa", name = "main", split="train")
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data['question_stem']
                    # self.question = [f'{ques.strip()} Knowledge: {know.strip()}' for ques, know in zip(questions, self.knowledges)]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                self.data = load_dataset("allenai/openbookqa", name = "main", split="test")
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data['question_stem']
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "siqa":
            if self.is_train:
                self.data = load_dataset("allenai/social_i_qa", split="train")
                answers = self.data["label"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                self.data = load_dataset("allenai/social_i_qa", split="validation")
                answers = self.data["label"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "wngr":
            if self.is_train:
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="train")
                answers = self.data["answer"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["sentence"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
                answers = self.data["answer"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["sentence"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "boolq":
            if self.is_train:
                self.data = load_dataset("google/boolq", split="train")
                answers = self.data["answer"]
            
                self.answers = [0 if answer else 1 for answer in answers]
                
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                self.data = load_dataset("google/boolq", split="validation")
                answers = self.data["answer"]
                self.answers = [0 if answer else 1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        # domain specific
        elif self.task == "acl_arc":
            label2id = {
                "background": 0,
                "motivation": 1,
                "compareorcontrast": 2,
                "uses": 3,
                "extends": 4,
                "future": 5,
            }
            
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, back_ratio_)
                
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
                questions = [item["text"] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
            self.answers = [label2id[line["label"].lower().strip()] for line in self.data]
        elif self.task == "chemprot":
            label2id = {
                "INHIBITOR": 0,
                "SUBSTRATE": 1,
                "INDIRECT-UPREGULATOR": 2,
                "INDIRECT-DOWNREGULATOR": 3,
                "ACTIVATOR": 4,
                "ANTAGONIST": 5,
                "PRODUCT-OF": 6,
                "AGONIST": 7,
                "DOWNREGULATOR": 8,
                "UPREGULATOR": 9,
                "AGONIST-ACTIVATOR": 10,
                "SUBSTRATE_PRODUCT-OF": 11,
                "AGONIST-INHIBITOR": 12,
            }
            
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, back_ratio_)
                
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
                questions = [item["text"] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question

            self.answers = [label2id[line["label"].upper().strip()] for line in self.data]
        elif self.task == "scicite":
            label2id = {
                "background": 0,
                "method": 1,
                "result": 2,
            }
            
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, back_ratio_)
                
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
                questions = [item["text"] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question

            self.answers = [label2id[line["label"].lower().strip()] for line in self.data]
        
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
            # if (overall_confidence >= ratio) and (not self.is_train): # inference # original
            if (overall_confidence >= ratio) and (overall_confidence <= self.upper_ratio)and (not self.is_train): # confidence 
                if ((cur_answer == None) or ("none" in cur_answer.lower()) or ("neither" in cur_answer.lower()) or duplicate(cur_answer)): 
                # if (cur_answer == None):
                    # print("this instance has null answer") # error
                    back_knowledge = background_processing(knowledge, back_ratio) 
                    try:
                        back_answer_knowledge = knowledge["Answer_Knowledge"][0]["Knowledge"] # answer knowledge
                        back_answer_knowledge = f'{back_knowledge}{knowledge["Answer_Knowledge"][0]["Knowledge"]}.'
                    except:
                        back_answer_knowledge = back_knowledge
                        
                    back_answer_knowledge = back_knowledge # only back
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
                self.knowledges.append(back_knowledge)
            else: # train or inference
                if self.is_train:
                    back_answer_knowledge = background_processing(knowledge, back_ratio) 
               
                else: # inference
                    back_knowledge = background_processing(knowledge, back_ratio)   
                    if cur_answer_knowledge_confidence >= back_ratio: # original
                    # if cur_answer_knowledge_confidence >= ratio: # only background knowledge
                        try:
                            back_answer_knowledge = f'{back_knowledge}{knowledge["Answer_Knowledge"][0]["Knowledge"]}.' # in stqa answer-answer -> 65.50, answer-back&answer -> 65.50%
                        except:
                            back_answer_knowledge = back_knowledge
                        
                    else:
                        back_answer_knowledge = back_knowledge # original
                    back_answer_knowledge = back_knowledge # original
                
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
            max_length=512 # ours
            # max_length=256 
        )
        
        return encoding
    
    def huggingface_formatting(self):
        """
        에러가 존재하여, ARC challgenge는 별도의 함수를 구축하여, 처리한다.
        """
        idx = 0
        prompts = []
        final_answer = []
        final_knowledges = []
        final_dir_answer = []
        for item in self.data:
            choices = item['choices']['text']
            if self.is_ours:
                question = self.question[idx]
            else:
                if self.task in ["arc-e", "arc-h"]:
                    question = item['question']
                elif self.task == "obqa":
                    question = item['question_stem']
            
            if self.task in ["arc-e", "arc-h"]:
                template = arc_instruction
            elif self.task == "obqa":
                template = obqa_instruction
                
            if len(choices) == 4:
                formatted_prompt = template.format(
                    question=question,
                    sol1=choices[0],
                    sol2=choices[1],
                    sol3=choices[2],
                    sol4=choices[3]
                )
                final_answer.append(self.answers[idx])
                if self.is_ours:
                    final_knowledges.append(self.knowledges[idx])
                    if not self.is_train:
                        final_dir_answer.append(self.answer[idx])
                prompts.append(preprocess(formatted_prompt, self.tokenizer.eos_token))
                idx += 1
            else:
                idx += 1
                continue
            
        self.answers = final_answer
        self.knowledges = final_knowledges
        if not self.is_train:
            self.answer = final_dir_answer
        return prompts
    
    def prepare_sft_dataset(self, verbose=True):
        if self.task == "piqa":
            if self.is_ours:
                prompts = [preprocess(piqa_instruction.format(question=ques, sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(piqa_instruction.format(question=item["goal"], sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "csqa":
            if self.is_ours:
                prompts = [preprocess(csqa_instruction.format(question=ques, sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(csqa_instruction.format(question=item['question']['stem'], sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "stqa":
            if self.is_ours:
                prompts = [preprocess(stqa_instruction.format(question=ques, sol1="True", sol2="False"), self.tokenizer.eos_token) for _, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(stqa_instruction.format(question=item['question'], sol1="True", sol2="False"), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "qasc":
            if self.is_ours:
                prompts = [preprocess(qasc_instruction.format(question=ques, sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(qasc_instruction.format(question=item['question']['stem'], sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task in ["arc-e", "arc-h", "obqa"]:
            prompts = self.huggingface_formatting()
            print(f'second # of answer: {len(self.answers)}')
        elif self.task == "siqa":
            if self.is_ours:
                prompts = [preprocess(siqa_instruction.format(context=item["context"], question=ques, sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(siqa_instruction.format(context=item["context"],question=item['question'], sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "wngr":
            if self.is_ours:
                prompts = [preprocess(wngr_instruction.format(question=ques, sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(wngr_instruction.format(question=item['sentence'], sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "boolq":
            if self.is_ours:
                prompts = [preprocess(boolq_instruction.format(context=item["passage"], question=ques), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(boolq_instruction.format(context=item["passage"], question=item['question']), self.tokenizer.eos_token) for item in self.data]          
        elif self.task == "chemprot":
            if self.is_ours:
                prompts = [preprocess(chemprot_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(chemprot_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]          
        elif self.task == "acl_arc":
            if self.is_ours:
                prompts = [preprocess(acl_arc_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(acl_arc_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]        
        elif self.task == "scicite":
            if self.is_ours:
                prompts = [preprocess(scicite_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(scicite_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]              
                
        
        self.labels = [int(item) for item in self.answers]
        self.encoding = self.tokenizing(prompts)
        
        # torch.set_printoptions(profile="full") # for code verifying
        if verbose:
            print(f'sample example: {prompts[0]}')
            print(self.labels[0])

        return None
    
    def validation(self):
        # inputs
        if self.task == "piqa":
            if self.is_ours:
                self.inputs = [preprocess(piqa_instruction.format(question=f'{item["goal"].strip()} {ques}', sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(piqa_instruction.format(question=item["goal"], sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "csqa":
            if self.is_ours:
                self.inputs = [preprocess(csqa_instruction.format(question=ques, sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(csqa_instruction.format(question=item['question']['stem'], sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "stqa":
            if self.is_ours:
                self.inputs = [preprocess(stqa_instruction.format(question=ques, sol1="True", sol2="False"), self.tokenizer.eos_token) for _, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(stqa_instruction.format(question=item['question'], sol1="True", sol2="False"), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "qasc":
            if self.is_ours:
                self.inputs = [preprocess(qasc_instruction.format(question=ques, sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(qasc_instruction.format(question=item['question']['stem'], sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task in ["arc-e", "arc-h", "obqa"]:
            self.inputs = self.huggingface_formatting()
            
            print(f'second # of answer: {len(self.answers)}')
        elif self.task == "siqa":
            if self.is_ours:
                self.inputs = [preprocess(siqa_instruction.format(context=item["context"], question=ques, sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(siqa_instruction.format(context=item["context"], question=item['question'], sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "wngr":
            if self.is_ours:
                self.inputs = [preprocess(wngr_instruction.format(context=item["context"], question=ques, sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(wngr_instruction.format(context=item["context"], question=item['sentence'], sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "boolq":
            if self.is_ours:
                self.inputs = [preprocess(boolq_instruction.format(context=item["passage"], question=ques), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(boolq_instruction.format(context=item["passage"], question=item['question']), self.tokenizer.eos_token) for item in self.data] 
        elif self.task == "chemprot":
            if self.is_ours:
                self.inputs = [preprocess(chemprot_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                self.inputs = [preprocess(chemprot_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]          
        elif self.task == "acl_arc":
            if self.is_ours:
                self.inputs = [preprocess(acl_arc_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                self.inputs = [preprocess(acl_arc_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]   
        elif self.task == "scicite":
            if self.is_ours:
                self.inputs = [preprocess(scicite_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                self.inputs = [preprocess(scicite_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]              
        
        # labels
        self.labels = [int(item) for item in self.answers]
        
        
        print(self.inputs[0])
        print(self.labels[0])
        return None
    
    def __getitem__(self, index):
        return {
            "input_ids": self.encoding["input_ids"][index],
            "attention_mask": self.encoding["attention_mask"][index],
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)  
        
    # def __getitem__(self, index):
    #     data = {key: val[index] for key, val in self.encoding.items()}
    #     data['labels'] = self.labels[index]
    #     return data 

    # def __len__(self) -> int:
    #     return len(self.labels)

if __name__ == "__main__":
    """"
    Instruction model에도 INST라는 토큰은 없다.
    """
    tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")
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

    print(tokenizer.convert_ids_to_tokens(50257))
    print(tokenizer.convert_ids_to_tokens(50259))
    print(tokenizer.eos_token)
    # tokenizer.padding_side = "right" # standard
    data = CorrectionJSONDataset("boolq", "./dataset/boolq/converted_train.json",
                                 "./dataset/boolq/converted_train.json", tokenizer, True,True, "./dataset/boolq/converted_train.json", ratio_value=0, back_ratio_value=0)
    print(data[0])
    
    
    
    # def postprocessing_prev(self, knowledges, ratio):
    #     self.answer_knowledge = [knowledge.get("Answer_Knowledge", knowledge.get("Background_Knowledge", "Only consider the given question")).strip() for knowledge in knowledges]
    #     self.back_knowledge = [knowledge.get("Background_Knowledge", knowledge.get("Anwer_Knowledge", "Only consider the given question")).strip() for knowledge in knowledges]
    #     self.confidence = [confidence.get("Confidence", 0) for confidence in knowledges]
    #     self.knowledges = []
        
    #     for idx in range(len(self.answer_knowledge)):
    #         if float(self.confidence[idx]) > ratio:
    #             if self.answer_knowledge[idx] == "Not_Knowledge":
    #                 self.knowledges.append(self.back_knowledge[idx])
    #             else: # related to answer
    #                 self.knowledges.append(self.answer_knowledge[idx])
    #         else:
    #             if self.back_knowledge[idx] == "Not_Knowledge":
    #                 self.knowledges.append(self.answer_knowledge[idx])
    #             else: # related to question
    #                 self.knowledges.append(self.back_knowledge[idx])