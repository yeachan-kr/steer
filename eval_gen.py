from open_source import Open_source_LM
from dataset_gen import Gen_Dataset
from transformers import AutoTokenizer
import torch 
import pandas as pd
import numpy as np
import os
import argparse
import re
from utils import extract_answer


def transform_zero_shot_foramt(task, zero_shot_preds):
    transformed_zero_shot_preds = []

    for pred in zero_shot_preds:
        if pred == "Not answer":
            transformed_zero_shot_preds.append(-1)
        else:
            transformed_zero_shot_preds.append(extract_answer(task, pred)) # string -> int

    return transformed_zero_shot_preds



def evaluate(model_name, task_name, is_ours, is_lora, is_generate, is_seq2seq, ratio, back_ratio):    
    # load model
    model = Open_source_LM(model_name, is_generate, is_lora, is_seq2seq, 0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if task_name in ["gsm8k", "aqua"]:
        cr_path = f"./dataset/{task_name}/converted_test.json"
    else:
        cr_path = f"./dataset/{task_name}/converted_dev.json"
    
    # 분석 코드.
    # is_ours = False
    # print("True -> False")
    
    # load dataset
    print(f"evaluating on {task_name} task")

    task_dataset = Gen_Dataset(task_name, question_path, answer_path, tokenizer, False, is_ours, cr_path, ratio, back_ratio)
    inputs = task_dataset.inputs
    gold_labels = task_dataset.labels
    gold_labels = [extract_answer(task_name, label) for label in gold_labels] # extracted labels
    
    if is_ours:
        zero_shot_preds = task_dataset.answer
        zero_shot_preds = transform_zero_shot_foramt(task_name, zero_shot_preds)
        
        zero_shot_standard = task_dataset.knowledges
    
    with torch.no_grad():
        preds = model.get_output(inputs)["pred._label"] # TYPE: pd
        preds = [extract_answer(task_name, pred) for pred in preds] # extracted predictions
     
        acc = 0
        changed_count = 0
        confi_preds = []
        confi_golds = []

        if is_ours:
            for idx in range(len(zero_shot_preds)):
                if zero_shot_standard[idx] == "Direct Answer":
                    changed_count += 1
                    confi_preds.append(zero_shot_preds[idx])
                    confi_golds.append(gold_labels[idx])
                    preds[idx] = zero_shot_preds[idx]
                    
            llm_zero_shot_acc = np.sum(np.where(np.array(confi_preds) == np.array(confi_golds), 1, 0)) *100 / len(confi_golds)
            print(f'our confidence zero-shot LLM: {llm_zero_shot_acc}')   
            print(f'changed {changed_count}')
        # classification task -> Acc.
        acc = np.sum(np.where(np.array(preds) == np.array(gold_labels), 1, 0)) *100 / len(gold_labels)
        
        print(f'ours acc.: {acc}')
        
    return acc, preds, gold_labels

    
if __name__ == "__main__":
    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--model_name', help='fine-tuned korean model: huggingface_model_path', type=str, required=True)
    parser.add_argument('--target_task', help='targeting fine-tuning dataset', required=True, type=str)
    parser.add_argument('--is_ours', help='ours vs baseline', required=True, type=str)
    parser.add_argument('--is_lora', help='Llama-3.2-3B', required=True, type=str)
    parser.add_argument('--seed', help='seed', required=True, type=str)
    parser.add_argument('--ratio', help='0~1', type=float, required=True)
    parser.add_argument('--back_ratio', help='0~1', type=float, required=True)
    # required = False
    
    args = parser.parse_args()
    ratio = args.ratio
    back_ratio = args.back_ratio
    seed = args.seed
    target_task = args.target_task
    model_name = args.model_name
    
    if target_task =="gsm8k":
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [467, 935, 1402, 1870, 2337, 2802]
    if target_task =="aqua":
        question_path = None
        answer_path = None
        ckpt = [6092, 12184, 18276, 24368, 30460, 36552]



    if args.is_ours == "yes":
        is_ours = True
    else:
        is_ours = False
    
    if args.is_lora == "yes":
        is_lora = True
        lora_True = "lora_True"
    else:
        is_lora = False
        lora_True = "lora_False"
    
    is_generate = True
    is_seq2seq = False
    
    # epoch   
    for idx in ckpt:
        current_model_name = f'./models/{target_task}/cls/{model_name}/{is_ours}/{lora_True}/{seed}/{back_ratio}/checkpoint-{idx}'
        # current_model_name = f'./models/{target_task}/cls/{model_name}/{is_ours}/{lora_True}/{seed}/{back_ratio}/checkpoint-{idx}'
        print(f'current_ratio: {ratio}')
        
        # current_model_name = f'./models/{target_task}/cls/{model_name}/False/{lora_True}/{seed}/checkpoint-{(idx+1)*ckpt}' # 분석 코드
        
        print(f'current model name is {current_model_name}')
        acc, preds_list, gold_labels = evaluate(current_model_name, target_task, is_ours, is_lora, is_generate, is_seq2seq, ratio, back_ratio)

        df = pd.DataFrame({"prediction": preds_list,
                        "labels": gold_labels,
                        "acc": acc})
        df.to_csv(f"./results/{model_name}/{target_task}/{seed}/{is_ours}/{lora_True}_{idx}.csv", header=True, index=False)
        # df.to_csv(f"./results/llama_confusion_matrix/{target_task}/{is_ours}/{lora_True}.csv", header=True, index=False)
        
        
        
        
        # 분석 코드. 실행 후 제거하기.
        # if target_task == "csqa":
        #     # current_model_name = f'./models/backbone_mistral/csqa/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-3045'
        #     current_model_name = f'./models/backbone_3B/csqa/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-1218'
        #     ratio = 0.9
        # elif target_task == "piqa":
        #     # current_model_name = f'./models/backbone_mistral/piqa/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-5040' # piqa
        #     current_model_name = f'./models/backbone_3B/piqa/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-6048'
        #     ratio = 0.9
        # elif target_task == "obqa":
        #     # current_model_name = f'./models/backbone_mistral/obqa/cls/Llama-3.2-1B-Instruct/True/lora_True/2/0.9/checkpoint-1240' # obqa
        #     current_model_name = f'./models/backbone_3B/obqa/cls/Llama-3.2-1B-Instruct/True/lora_True/2/0.9/checkpoint-1240' # obqa
        #     ratio = 0.9
        # elif target_task == "arc-h":
        #     # current_model_name = f'./models/backbone_mistral/arc-h/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-1865' # arc-h
        #     current_model_name = f'./models/backbone_3B/arc-h/cls/Llama-3.2-1B-Instruct/True/lora_True/2/0.9/checkpoint-1492' # arc-h
        #     ratio = 0.7
        # elif target_task == "qasc":
        #     # current_model_name = f'./models/backbone_mistral/qasc/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-2036'
        #     current_model_name = f'./models/backbone_3B/qasc/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-3054'
        #     ratio = 0.9
        # elif target_task == "stqa":
        #     current_model_name = f'./models/backbone_mistral/stqa/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-516'
        #     # current_model_name = f'./models/backbone_3B/stqa/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-516'
        #     ratio = 0.9
        
        
        
        # if target_task == "csqa":
        #     current_model_name = f'./models/backbone_3B/csqa/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-1218'
        #     ratio = 0.7
        # elif target_task == "piqa":
        #     # current_model_name = f'./models/backbone_mistral/piqa/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-5040' # piqa
        #     current_model_name = f'./models/backbone_3B/piqa/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-6048'
        #     ratio = 0.7
        # elif target_task == "obqa":
        #     # current_model_name = f'./models/backbone_mistral/obqa/cls/Llama-3.2-1B-Instruct/True/lora_True/2/0.9/checkpoint-1240' # obqa
        #     current_model_name = f'./models/backbone_3B/obqa/cls/Llama-3.2-1B-Instruct/True/lora_True/2/0.9/checkpoint-1240' # obqa
        #     ratio = 0.9
        # elif target_task == "arc-h":
        #     # current_model_name = f'./models/backbone_mistral/arc-h/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-1865' # arc-h
        #     current_model_name = f'./models/backbone_3B/arc-h/cls/Llama-3.2-1B-Instruct/True/lora_True/2/0.9/checkpoint-1492' # arc-h
        #     ratio = 0.7
        # elif target_task == "qasc":
        #     # current_model_name = f'./models/backbone_mistral/qasc/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-2036'
        #     current_model_name = f'./models/backbone_3B/qasc/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-3054'
        #     ratio = 0.5
        # elif target_task == "stqa":
        #     # current_model_name = f'./models/backbone_mistral/stqa/cls/Llama-3.2-1B-Instruct/True/lora_True/42/0.9/checkpoint-516'
        #     current_model_name = f'./models/backbone_3B/stqa/cls/Llama-3.2-1B-Instruct/True/lora_True/10/0.9/checkpoint-516'
        #     ratio = 0.9