from open_source import CombLM
from dataset_proxy import ProxyDataset
from transformers import AutoTokenizer
import torch 
import pandas as pd
import numpy as np
import os
import argparse
import re
from utils import proxy_answer




def evaluate(base, expert, is_constant, task_name, alpha_list):    
    # load model
    for alpha in alpha_list:
        model = CombLM(base, expert, is_constant, task_name, alpha)
        tokenizer = AutoTokenizer.from_pretrained(expert)
        print(f"evaluating on {task_name} task")
        task_dataset = ProxyDataset(task_name, question_path, answer_path, tokenizer, False)
        inputs = task_dataset.inputs
        gold_labels = task_dataset.labels

        with torch.no_grad():
            preds = model.get_output(inputs)["pred._label"] # TYPE: pd
            preds = [proxy_answer(pred, task_name) for pred in preds] # extracted predictions
            # print(preds)
            acc = 0
            acc = np.sum(np.where(np.array(preds) == np.array(gold_labels), 1, 0)) *100 / len(gold_labels)
            
            print(f'ours acc.: {acc}')
        

    
if __name__ == "__main__":
    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--model_name', help='fine-tuned korean model: huggingface_model_path', type=str, required=True)
    parser.add_argument('--target_task', help='targeting fine-tuning dataset', required=True, type=str)
    parser.add_argument('--seed', help='seed', required=True, type=str)
    # required = False
    parser.add_argument('--is_constant', help='mean, constant_scalar', type=str, required=False, default="no")
    args = parser.parse_args()
    seed = args.seed
    target_task = args.target_task
    model_name = args.model_name
    
    if args.is_constant == 'yes':
        is_constant = True
        alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    elif args.is_constant == 'no':
        is_constant = False
        
    
    
    if target_task == "piqa":
        question_path = "./dataset/piqa/dev.jsonl"
        answer_path = "./dataset/piqa/dev-labels.lst"
        ckpt = [3021]
    elif target_task == "csqa":
        question_path = "./dataset/csqa/dev_rand_split.jsonl"
        answer_path = "./dataset/csqa/dev_rand_split.jsonl"
        ckpt = [1218]
    elif target_task == "qasc":
        question_path = "./dataset/qasc/dev.jsonl"
        answer_path = "./dataset/qasc/dev.jsonl"
        ckpt = [1018]
    elif target_task =="arc-h":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [1118]
    elif target_task =="obqa":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [620]
    elif target_task =="wngr":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [7575]
    elif target_task =="siqa":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [8353]
    elif target_task =="stqa":
        question_path = f"./dataset/{target_task}/dev.json"
        answer_path = f"./dataset/{target_task}/dev.json"
        ckpt = [258]
    elif target_task =="boolq":
        question_path = f"./dataset/{target_task}/dev.json"
        answer_path = f"./dataset/{target_task}/dev.json"
        ckpt = [1180]

    
    # epoch   
    for idx in ckpt:
        base_name = "meta-llama/Llama-3.2-3B-Instruct"
        current_model_name = f'./models/proxy/{target_task}/cls/{model_name}/lora_True/{seed}/checkpoint-{idx}'
        print(f'current model name is {current_model_name}')
        evaluate(base_name, current_model_name, is_constant, target_task, alpha_list)

        
        
