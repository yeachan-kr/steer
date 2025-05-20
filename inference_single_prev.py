from transformers import pipeline
import torch
from utils import SYSTEM_HEAD, SYSTEM_CONTENT, USER_HEAD, USER_CONTENT, ASISTANT_HEAD
import jsonlines
import re
import pandas as pd
import generate_template
import json
import argparse
import torch
torch.cuda.empty_cache()
from datasets import load_dataset

##just tags##
def annotate_tag(r):
    r = r.strip()
    if any(x in r for x in ['[[1]]', '[1]', '1']):
        if any(x in r for x in ['[[2]]', '[2]', '2']):
            return r # 1 and 2 (both)
        if any(x in r for x in ['[[1]]', '[1]', '1']):
            return 1 # only 1
    
    if any(x in r for x in ['[[2]]', '[2]', '2']):
        return 2
    else:
        return r

def parse_background_knowledge(text: str) -> dict:    
    def _normalize(s: str) -> str:
        s = re.sub(r'\n(?!\s*\n)', ' ', s)
        s = re.sub(r'[ \t]+', ' ', s)
        return s.strip()

    text = _normalize(text)


    bg_pattern = re.compile(
        r'Background[_\s]+Knowledge[_\s]*(\d+)\s*:\s*'          
        r'(.*?)\s*'                                             
        r'(?:\(|,)?\s*confidence[:\s]*([0-9]{1,3})\s*%?\s*\)?', # confidence
        re.IGNORECASE | re.DOTALL
    )

    ans_pattern = re.compile(
        r'Answer[_\s]+Knowledge\s*:\s*'
        r'(.*?)\s*'
        r'(?:\(|,)?\s*confidence[:\s]*([0-9]{1,3})\s*%?\s*\)?',
        re.IGNORECASE | re.DOTALL
    )

    final_answer_pattern = re.compile(
        r'Final[_\s]+Answer\s*:\s*(.+?)(?=\s{2,}|\n{2,}|\Z)',
        re.IGNORECASE | re.DOTALL
    )

    final_confidence_pattern = re.compile(
        r'Overall[_\s]+Confidence.*?:\s*([0-9]{1,3})\s*%?',
        re.IGNORECASE
    )

   
    out = {
        "Background_Knowledge": [],
        "Answer_Knowledge": [],
        "Final_Answer": None,
        "Final_Confidence": None,
    }

    for num, know, conf in bg_pattern.findall(text):
        out["Background_Knowledge"].append({
            f"Knowledge_{num}": know.strip(),
            "Confidence": int(conf),
        })

    ans = ans_pattern.search(text)
    if ans:
        out["Answer_Knowledge"].append({
            "Knowledge": ans.group(1).strip(),
            "Confidence": int(ans.group(2)),
        })

    final_ans = final_answer_pattern.search(text)
    if final_ans:
        out["Final_Answer"] = final_ans.group(1).strip()

    overall = final_confidence_pattern.search(text)
    if overall:
        out["Final_Confidence"] = int(overall.group(1))

    return out

if __name__ == "__main__":
    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--task', help='piqa, csqa, stqa, qasc', type=str, required=True)
    parser.add_argument('--type', help='train or dev', type=str, required=True)
    # required = False
    parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=False, default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    data_type = args.type
    model_ckpt = args.model_ckpt
    task = args.task
    
    data = []
    if task == "piqa":
        if data_type == "train":
            question_path = "./dataset/piqa/train.jsonl"
            answer_path = "./dataset/piqa/train-labels.lst"
        else:
            question_path = "./dataset/piqa/dev.jsonl"
            answer_path = "./dataset/piqa/dev-labels.lst"
        # input
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        # label
        with open(answer_path, 'r') as file:
            answers = [line.strip() for line in file.readlines()]
    elif task == "csqa":
        if data_type == "train":
            question_path = "./dataset/csqa/train_rand_split.jsonl"
        else:
            question_path = './dataset/csqa/dev_rand_split.jsonl'
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in data]
    elif task == "stqa":
        if data_type == "train":
            question_path = "./dataset/stqa/train.json"
        else:
            question_path = "./dataset/stqa/dev.json"
        # input
        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        answers = [1 if line["answer"] else 0 for line in data]
    elif task == "qasc":
        if data_type == "train":
            question_path = "./dataset/qasc/train.jsonl"
        else:
            question_path = "./dataset/qasc/dev.jsonl"
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                    if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                    if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                    if line['answerKey'] == "G" else 7 for line in data]
    
    elif task == "arc-e":
        if data_type == "train":
            data = load_dataset("allenai/ai2_arc", name = "ARC-Easy", split="train")
        else:
            data = load_dataset("allenai/ai2_arc", name = "ARC-Easy", split="validation")

        answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in data["answerKey"]]
     
    
    pipe = pipeline(
            "text-generation", 
            model=model_ckpt, 
            torch_dtype=torch.bfloat16, 
            max_new_tokens=256,
            device_map="auto",
            do_sample=False
        )
    
    outputs = []
    for instance in data:
        if task == "piqa":
            question = instance["goal"]
            sol1 = instance["sol1"]
            sol2 = instance["sol2"]
            prompt = generate_template.piqa_template.format(goal=question, sol1=sol1, sol2=sol2) # piqa
            # prompt = generate_template.gkp_template_piqa_0327.format(goal=question, sol1=sol1, sol2=sol2) # piqa
        elif task == "csqa": 
            question = instance['question']['stem']
            sol1 = instance["question"]["choices"][0]['text']
            sol2 = instance["question"]["choices"][1]['text']
            sol3 = instance["question"]["choices"][2]['text']
            sol4 = instance["question"]["choices"][3]['text']
            sol5 = instance["question"]["choices"][4]['text']
            prompt = generate_template.csqa_template.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)
            # prompt = generate_template.gkp_template_csqa_0327.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5) # csqa
        elif task == "stqa":
            question = instance['question']
            prompt = generate_template.stqa_template.format(goal=question)
            # prompt = generate_template.gkp_template_stqa_0327.format(goal=question) # stqa
        elif task == "qasc":
            question = instance['question']['stem']
            sol1 = instance['question']['choices'][0]['text']
            sol2 = instance["question"]["choices"][1]['text']
            sol3 = instance["question"]["choices"][2]['text']
            sol4 = instance["question"]["choices"][3]['text']
            sol5 = instance["question"]["choices"][4]['text']
            sol6 = instance["question"]["choices"][5]['text']
            sol7 = instance["question"]["choices"][6]['text']
            sol8 = instance["question"]["choices"][7]['text']
            prompt = generate_template.qasc_template.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5, sol6=sol6, sol7=sol7, sol8=sol8) # qasc
            # prompt = generate_template.gkp_template_qasc_0327.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5, sol6=sol6, sol7=sol7, sol8=sol8) # qasc
        
        
        
        split_point = len(prompt)
        print(f'original input: {question}')
        output = pipe(prompt)[0]['generated_text'][split_point:]
        print(output)
        output = parse_background_knowledge(output)
        print(output)
        outputs.append(output)

    with open(f"./dataset/{task}/converted_{data_type}.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)   
