from open_source import Dexpert
from dataset_proxy import ProxyDataset
from transformers import AutoTokenizer
import torch 
import pandas as pd
import numpy as np
import os
import argparse
import re
from utils import proxy_answer




def evaluate(base, expert, antiexpert, task_name):    
    # load model
    model = Dexpert(base, expert, antiexpert, task_name)
    tokenizer = AutoTokenizer.from_pretrained(expert)
    print(f"evaluating on {task_name} task")
    task_dataset = ProxyDataset(task_name, question_path, answer_path, tokenizer, False)
    inputs = task_dataset.inputs
    gold_labels = task_dataset.labels

    with torch.no_grad():
        preds = model.get_output(inputs)["pred._label"] # TYPE: pd
        preds = [proxy_answer(pred, task_name) for pred in preds] # extracted predictions
        print(preds)
        acc = 0
        acc = np.sum(np.where(np.array(preds) == np.array(gold_labels), 1, 0)) *100 / len(gold_labels)
        
        print(f'ours acc.: {acc}')
        
    return acc, preds, gold_labels

    
if __name__ == "__main__":
    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--model_name', help='fine-tuned korean model: huggingface_model_path', type=str, required=True)
    parser.add_argument('--target_task', help='targeting fine-tuning dataset', required=True, type=str)
    parser.add_argument('--seed', help='seed', required=True, type=str)
    # required = False
    
    args = parser.parse_args()
    seed = args.seed
    target_task = args.target_task
    model_name = args.model_name
    
    if target_task == "piqa":
        question_path = "./dataset/piqa/dev.jsonl"
        answer_path = "./dataset/piqa/dev-labels.lst"
        ckpt = [1007, 2014, 3021, 4029, 5036, 6042]
    elif target_task == "csqa":
        question_path = "./dataset/csqa/dev_rand_split.jsonl"
        answer_path = "./dataset/csqa/dev_rand_split.jsonl"
        ckpt = [609,1218,1827,2436,3045,3654]
    elif target_task == "qasc":
        question_path = "./dataset/qasc/dev.jsonl"
        answer_path = "./dataset/qasc/dev.jsonl"
        ckpt = [509*idx for idx in range(1,7)]
    elif target_task =="arc-h":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [559*idx for idx in range(1,7)]
    elif target_task =="obqa":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [310*idx for idx in range(1,7)]
    elif target_task =="wngr":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [2525*idx for idx in range(1,7)]
    elif target_task =="siqa":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [2088, 4176, 6264, 8353, 10441, 12528]
    elif target_task =="stqa":
        question_path = f"./dataset/{target_task}/dev.json"
        answer_path = f"./dataset/{target_task}/dev.json"
        ckpt = [129*idx for idx in range(1,7)]
    elif target_task =="boolq":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [1180]

    
    # epoch   
    for idx in ckpt:
        base_name = "meta-llama/Llama-3.2-3B-Instruct"
        slm_name = "meta-llama/Llama-3.2-1B-Instruct"
        current_model_name = f'./models/proxy/{target_task}/cls/{model_name}/lora_True/{seed}/checkpoint-{idx}'
        print(f'current model name is {current_model_name}')
        acc, preds_list, gold_labels = evaluate(base_name, current_model_name, slm_name, target_task)

        
        
