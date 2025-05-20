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
import numpy as np
from collections import Counter
from datasets import load_dataset
import random
from transformers.utils import logging
logging.set_verbosity_error()

random.seed(42)


if __name__ == "__main__":
    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--task', help='piqa, csqa, stqa, qasc', type=str, required=True)
    parser.add_argument('--baseline', help='gkp, selftalk', type=str, required=True)
    # required = False
    parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=False, default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    model_ckpt = args.model_ckpt
    task = args.task
    baseline = args.baseline
    data = []
    if task == "piqa":
        question_path = "./dataset/piqa/dev.jsonl"
        answer_path = "./dataset/piqa/dev-labels.lst"
        if baseline == "selftalk":
            knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_piqa_dev.json"
        elif baseline == "gkp":
            knowledge_path = "./knowledges/gkp_llama_3b/knowledge_gkp_piqa_dev.json"
        # input
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        # label
        with open(answer_path, 'r') as file:
            answers = [int(line) for line in file.readlines()]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "csqa":
        question_path = './dataset/csqa/dev_rand_split.jsonl'
        if baseline == "selftalk":
            knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_csqa_dev.json"
        elif baseline == "gkp":
            knowledge_path = "./knowledges/gkp_llama_3b/knowledge_gkp_csqa_dev.json"
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in data]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "stqa":
        question_path = './dataset/stqa/dev.json'
        knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_stqa_dev.json"

        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        answers = [0 if line["answer"] else 1 for line in data] # True -> 0 (A), False -> 1 (B)
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "qasc":
        question_path = "./dataset/qasc/dev.jsonl"
        if baseline == "selftalk":
            knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_qasc_dev.json"
        elif baseline == "gkp":
            knowledge_path = "./knowledges/gkp_llama_3b/knowledge_gkp_qasc_dev.json"
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                    if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                    if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                    if line['answerKey'] == "G" else 7 for line in data]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "arc-h":
        if baseline == "selftalk":
            knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_arc_h_test.json"
        elif baseline == "gkp":
            knowledge_path = "./knowledges/gkp_llama_3b/knowledge_gkp_arc_h_test.json"
        arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                             '1': 0, '2': 1, '3': 2, '4': 3} # 정답값이 이상하게 배치되어 있어서.
        data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
        answers = data["answerKey"]
        answers = [arc_label_map[str(answer)] for answer in answers]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "obqa":
        if baseline == "selftalk":
            knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_obqa_test.json"
        elif baseline == "gkp":
            knowledge_path = "./knowledges/gkp_llama_3b/knowledge_gkp_obqa_test.json"
        data = load_dataset("allenai/openbookqa", name = "main", split="test")
        answers = data["answerKey"]
        answers = [0 if answer == "A" else 1 if answer == "B" else 2 if answer == "C" else 3 for answer in answers]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "siqa":
        if baseline == "selftalk":
            knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_siqa_dev.json"
        elif baseline == "gkp":
            knowledge_path = "./knowledges/gkp_llama_3b/knowledge_gkp_siqa_dev.json"
        data = load_dataset("allenai/social_i_qa", split="validation")
        answers = data["label"]
        answers = [int(answer)-1 for answer in answers]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "wngr":
        if baseline == "selftalk":
            knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_wg_dev.json"
        elif baseline == "gkp":
            knowledge_path = "./knowledges/gkp_llama_3b/knowledge_gkp_wg_dev.json"
        data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
        answers = data["answer"]
        answers = [int(answer)-1 for answer in answers]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    elif task == "boolq":
        knowledge_path = "./knowledges/selftalk_llama_3b/knowledge_selftalk_boolq_dev.json"
        data = load_dataset("google/boolq", split="validation")
        answers = data["answer"]
        answers = [0 if answer else 1 for answer in answers]
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledges_group = json.load(f)
    
    
    ### evaluate ###
    pipe = pipeline(
        "text-generation", 
        model=model_ckpt, 
        torch_dtype=torch.bfloat16, 
        max_new_tokens=16,
        device_map="auto",
        do_sample=False
    )
    
    idx = 0
    outputs = []
    for knowledges in knowledges_group:
        current_preds = []
        knowledges = knowledges["knowledges"]
        
        if baseline == "selftalk":
            knowledges = random.sample(knowledges, k=4) # 4개의 지식만 고려. 
            
        for knowledge in knowledges:
            if task == "piqa":
                prompt = f"Question: {data[idx]['goal']}\nChoices:\nA. {data[idx]['sol1']}\nB. {data[idx]['sol2']}\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # PIQA
            elif task == "stqa":
                prompt = f"Question: {data[idx]['question']}\nChoices:\nA. True\nB. False\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # QASC
            elif task == "boolq":
                prompt = f"Passage: {data[idx]['passage']}\nQuestion: {data[idx]['question']}\nChoices:\nA. True\nB. False\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # QASC
            elif task == "csqa":
                question = data[idx]['question']['stem']
                sol1 = data[idx]["question"]["choices"][0]['text']
                sol2 = data[idx]["question"]["choices"][1]['text']
                sol3 = data[idx]["question"]["choices"][2]['text']
                sol4 = data[idx]["question"]["choices"][3]['text']
                sol5 = data[idx]["question"]["choices"][4]['text']
                prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # CSQA
             
            elif task == "qasc":
                question = data[idx]['question']['stem']
                sol1 = data[idx]['question']['choices'][0]['text']
                sol2 = data[idx]["question"]["choices"][1]['text']
                sol3 = data[idx]["question"]["choices"][2]['text']
                sol4 = data[idx]["question"]["choices"][3]['text']
                sol5 = data[idx]["question"]["choices"][4]['text']
                sol6 = data[idx]["question"]["choices"][5]['text']
                sol7 = data[idx]["question"]["choices"][6]['text']
                sol8 = data[idx]["question"]["choices"][7]['text']
                prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nF. {sol6}\nG. {sol7}\nH. {sol8}\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # QASC
            
            elif task == "arc-h":
                question = data[idx]['question']               
                choices = data[idx]['choices']['text']
                sol1=choices[0] if len(choices) > 0 else ""
                sol2=choices[1] if len(choices) > 1 else ""
                sol3=choices[2] if len(choices) > 2 else ""
                sol4=choices[3] if len(choices) > 3 else ""
                prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is ("
            elif task == "obqa":
                question = data[idx]['question_stem']            
                choices = data[idx]['choices']['text']
                sol1=choices[0] if len(choices) > 0 else ""
                sol2=choices[1] if len(choices) > 1 else ""
                sol3=choices[2] if len(choices) > 2 else ""
                sol4=choices[3] if len(choices) > 3 else ""
                prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is ("
            elif task == "siqa":
                context = data[idx]["context"]
                question = data[idx]["question"]
                sol1=data[idx]["answerA"]
                sol2=data[idx]["answerB"]
                sol3=data[idx]["answerC"]
                prompt = f"Context: {context}\nQuestion: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is ("
            elif task == "wngr":
                question = data[idx]["sentence"] 
                sol1 = data[idx]["option1"]
                sol2 = data[idx]["option2"]
                prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nKnowledge: {knowledge}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is ("   
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Given a question, choose the single, most likely answer"},
                {"role": "user", "content": prompt}
            ]
            
            output = pipe(messages)[0]["generated_text"][-1]['content']
            
            for pred in output:
                if pred == "A":
                    current_preds.append(0)
                    break
                elif pred == "B":
                    current_preds.append(1)
                    break
                elif pred == "C":
                    current_preds.append(2)
                    break
                elif pred == "D":
                    current_preds.append(3)
                    break
                elif pred == "E":
                    current_preds.append(4)
                    break
                elif pred == "F":
                    current_preds.append(5)
                    break
                elif pred == "G":
                    current_preds.append(6)
                    break
                elif pred == "H":
                    current_preds.append(7)
                    break
            else:
                current_preds.append(-1)
        if idx == 0:
            print(prompt)
        # print(current_preds)
        count = Counter(current_preds)
        most_common = count.most_common(1)[0][0]
        outputs.append(int(most_common))
        idx += 1
    acc = np.sum(np.where(np.array(outputs) == np.array(answers), 1, 0)) *100 / len(answers)
    print(f'{baseline} {task} zero-shot acc: {acc}')
    

