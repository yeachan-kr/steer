import torch
import jsonlines
import json
import argparse
from openai import OpenAI
import torch
torch.cuda.empty_cache()
import numpy as np
import os
import time
from datasets import load_dataset
from utils import extract_answer, ps_plus_rea_prompt, ps_rea_prompt, cot_rea_prompt, answer_prompt


# before
def chat_completion(prompt):
    for _ in range(API_MAX_RETRY):
        try:
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
              model=model_ckpt,
              messages=[
                    {"role": "user", "content": prompt},
                ],
              temperature=0,
              top_p=1
            )
            
            response = response.choices[0].message.content
            return response
        
        except Exception as e:
            if 'policy' in str(e):
                print("Skipping due to openai policy")
                return 'None'
            print(type(e), e)
            print("trying again")
            time.sleep(API_RETRY_SLEEP)

    return "None"




if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY") 
    # API setting constants
    API_MAX_RETRY = 5
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"


    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--task', help='piqa, csqa, stqa, qasc', type=str, required=True)
    parser.add_argument('--baseline', help='zs_cot, zs_ps_plus, zs_ps', type=str, required=True)
    # required = False
    parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=False, default="gpt-3.5-turbo")
    args = parser.parse_args()

    if args.baseline == "zs_cot":
        rea_prompt = cot_rea_prompt
    elif args.baseline == "zs_ps":
        rea_prompt = ps_rea_prompt
    elif args.baseline == "zs_ps_plus":
        rea_prompt = ps_plus_rea_prompt
    
    model_ckpt = args.model_ckpt
    task = args.task
    
    data = []
    if task == "piqa":
        question_path = "./dataset/piqa/dev.jsonl"
        answer_path = "./dataset/piqa/dev-labels.lst"
        # input
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        # label
        with open(answer_path, 'r') as file:
            answers = [int(line) for line in file.readlines()]
    elif task == "csqa":
        question_path = './dataset/csqa/dev_rand_split.jsonl'
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in data]
    elif task == "stqa":
        question_path = "./dataset/stqa/dev.json"
        # input
        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        answers = [0 if line["answer"] else 1 for line in data]
    elif task == "qasc":
        question_path = "./dataset/qasc/dev.jsonl"
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                    if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                    if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                    if line['answerKey'] == "G" else 7 for line in data]
    elif task == "arc-h":  
        arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
        data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
        answers = data["answerKey"]
        answers = [arc_label_map[str(answer)] for answer in answers]
    elif task == "wngr":
        data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
        answers = data["answer"]
        answers = [int(answer)-1 for answer in answers]
    elif task == "obqa":
        data = load_dataset("allenai/openbookqa", name = "main", split="test")
        answers = data["answerKey"]
        answers = [0 if answer == "A" else 1 if answer == "B" else 2 if answer == "C" else 3 for answer in answers]
    elif task == "siqa":
        data = load_dataset("allenai/social_i_qa", split="validation")
        answers = data["label"] # 1,2,3
        answers = [int(answer)-1 for answer in answers] # 0,1,2 로 changing!! 
    elif task == "boolq":
        data = load_dataset("google/boolq", split="validation")
        answers = data["answer"]
        answers = [0 if answer else 1 for answer in answers]
    elif task == "gsm8k":
        data = load_dataset("openai/gsm8k", name = "main", split="test")
        answers = data["answer"]
        answers = [extract_answer(task, label) for label in answers]
    
    
    ### evaluate ###
    outputs = []
    
    for instance in data:
        inter_ans = "{rationale}\n{ans_prompt}"
        if task == "piqa":
            prompt_rea = f"Question: {instance['goal']}\nChoices:\nA. {instance['sol1']}\nB. {instance['sol2']}\nAssistant: {rea_prompt}" # PIQA
        
        elif task == "stqa":
            prompt_rea = f"Question: {instance['question']}\nChoices:\nA. True\nB. False\nAssistant: {rea_prompt}"
        
        elif task == "csqa":
            question = instance['question']['stem']
            sol1 = instance["question"]["choices"][0]['text']
            sol2 = instance["question"]["choices"][1]['text']
            sol3 = instance["question"]["choices"][2]['text']
            sol4 = instance["question"]["choices"][3]['text']
            sol5 = instance["question"]["choices"][4]['text']
            prompt_rea = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nAssistant: {rea_prompt}" # CSQA
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
            prompt_rea = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nF. {sol6}\nG. {sol7}\nH. {sol8}\nAssistant: {rea_prompt}" # QASC
        elif task == "obqa":
            question = instance['question_stem']
            choices = instance['choices']['text']
            sol1 = choices[0] if len(choices) > 0 else ""
            sol2 = choices[1] if len(choices) > 1 else ""
            sol3 = choices[2] if len(choices) > 2 else ""
            sol4 = choices[3] if len(choices) > 3 else ""
            prompt_rea = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nAssistant: {rea_prompt}"
        elif task == "arc-h":
            question = instance['question']
            choices = instance['choices']['text']
            sol1 = choices[0] if len(choices) > 0 else ""
            sol2 = choices[1] if len(choices) > 1 else ""
            sol3 = choices[2] if len(choices) > 2 else ""
            sol4 = choices[3] if len(choices) > 3 else ""
            prompt_rea = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nAssistant: {rea_prompt}"
        elif task == "wngr":
            question = instance['sentence']
            sol1 = instance["option1"]
            sol2 = instance["option2"]
            prompt_rea = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nAssistant: {rea_prompt}"
        elif task == "siqa":
            question = instance['question']
            sol1=instance["answerA"]
            sol2=instance["answerB"]
            sol3=instance["answerC"]
            context=instance["context"] # ref. unicorn and siqa paper.
            prompt_rea = f"Context: {context}\nQuestion: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nAssistant: {rea_prompt}"
        elif task == "boolq":
            question = instance['question']
            context=instance["passage"] # ref. boolq paper
            prompt_rea = f"Context: {context}\nQuestion: {question}\nChoices:\nA. True\nB. False\nAssistant: {rea_prompt}" 
        # generation
        elif task == "gsm8k":
            question = instance["question"]
            prompt_rea = f"Question: {question}\nAssistant: {rea_prompt}"

        output = chat_completion(prompt_rea)
        # print(output)
        cur_ans = inter_ans.format(rationale=output.strip(), ans_prompt=answer_prompt(task))
        prompt_ans = prompt_rea + "\n" + cur_ans # self-augmented 
        output = chat_completion(prompt_ans)
        print(output)
        
        anot = output.lower().split()
        if ("neither" in anot) or ("none" in anot):
            print("predicts neither answer")
            outputs.append(-1)
            continue
        
        # QA format
        if task in ["gsm8k"]:
            outputs.append(extract_answer(task, output))
        else:
            for pred in output:
                if pred == "A":
                    outputs.append(0)
                    break
                elif pred == "B":
                    outputs.append(1)
                    break
                elif pred == "C":
                    outputs.append(2)
                    break
                elif pred == "D":
                    outputs.append(3)
                    break
                elif pred == "E":
                    outputs.append(4)
                    break
                elif pred == "F":
                    outputs.append(5)
                    break
                elif pred == "G":
                    outputs.append(6)
                    break
                elif pred == "H":
                    outputs.append(7)
                    break
            else:
                outputs.append(-1)
        
 
    acc = np.sum(np.where(np.array(outputs) == np.array(answers), 1, 0)) *100 / len(answers)
    print(f'chatgpt {task} {args.baseline} acc: {acc}')
    

