from open_source import Open_source_LM
from dataset import CorrectionJSONDataset
from transformers import AutoTokenizer
import torch 
import pandas as pd
import numpy as np
import os
import argparse
from baseline_dataset import BaselineDataset
import evaluate
from sklearn.preprocessing import LabelBinarizer     # multi-class → one-hot

# option; if controliing output, check False
def clean_text(pred):
    if len(pred) >= 2:
        try:
            pred = pred.split("\n")[1]
            pred = pred.strip()
            
            return str(pred)
        except:
            return str(pred)
        
    return pred

def mapping(pred):
    for item in pred:
        if item == "A":
            return 0 
        elif item == "B":
            return 1
        elif item == "C":
            return 2
        elif item == "D":
            return 3
        elif item == "E":
            return 4
        elif item == "F":
            return 5
        elif item == "G":
            return 6
        elif item == "H":
            return 7



def transform_zero_shot_foramt(task, zero_shot_preds):
    transformed_zero_shot_preds = []
    if task == "stqa":
        for pred in zero_shot_preds:
            if pred == "Not answer":
                transformed_zero_shot_preds.append(-1)
            else:
                for item in pred:
                    if item == "A":
                        transformed_zero_shot_preds.append(1)
                        break
                    elif item == "B":
                        transformed_zero_shot_preds.append(0)
                        break
    
    else: # otherwise task
        for pred in zero_shot_preds:
            if pred == "Not answer":
                transformed_zero_shot_preds.append(-1)
            else:
                transformed_zero_shot_preds.append(mapping(pred)) # string -> int
        

    return transformed_zero_shot_preds

def text_acc(preds, is_generate):
    """
    text2class로 구현하는 방법의 역
    tokenizer는 대소문자 구분 안하기 때문에, lower() 적용. (생성에 영향 X.)
    """
    cr_pred_list = []
    error_count = 0
    for idx in range(len(preds)):
        if is_generate:
            cr_pred = str(preds[idx].strip())
            cr_pred = clean_text(cr_pred)
            if cr_pred == "0":
                cr_pred_list.append(0)
            elif cr_pred == "1":
                cr_pred_list.append(1)
            elif cr_pred == "2":
                cr_pred_list.append(2)
            elif cr_pred == "3":
                cr_pred_list.append(3)
            elif cr_pred == "4":
                cr_pred_list.append(4)
            elif cr_pred == "5":
                cr_pred_list.append(5)
            elif cr_pred == "6":
                cr_pred_list.append(6)
            elif cr_pred == "7":
                cr_pred_list.append(7)
            else:
                cr_pred_list.append(-100)
                print(cr_pred)
                error_count += 1
        else:
            if int(preds[idx]) < 15:
                cr_pred_list.append(int(preds[idx]))
            else:
                cr_pred_list.append(-100)
                print(int(preds[idx]))
                error_count += 1
    print(f'# of error: {error_count}')
    return cr_pred_list

def evaluater(model_name, task_name, is_ours, is_lora, is_generate, is_seq2seq, num_labels, ratio, back_ratio, upper_ratio):    
    # load model
    model = Open_source_LM(model_name, is_generate, is_lora, is_seq2seq, num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if task_name in ["arc-h", "obqa", "chemprot", "acl_arc"]:
        cr_path = f"./dataset/{task_name}/converted_test.json"
    else:
        cr_path = f"./dataset/{task_name}/converted_dev.json"
    
    # 분석 코드.
    # is_ours = False
    # print("True -> False")
    
    # load dataset
    print(f"evaluating on {task_name} task")
    
    task_dataset = CorrectionJSONDataset(task_name, question_path, answer_path, tokenizer, False, is_ours, cr_path, ratio, back_ratio, upper_ratio)
    # task_dataset = BaselineDataset(task_name, question_path, answer_path, tokenizer,False, None, "gkp") # fine-tuning w/ baseline
    inputs = task_dataset.inputs
    gold_labels = task_dataset.labels
    
        
    if is_ours:
        zero_shot_preds = task_dataset.answer
        
        zero_shot_preds = transform_zero_shot_foramt(task_name, zero_shot_preds)
      
        zero_shot_standard = task_dataset.knowledges
        
    with torch.no_grad():
        preds = model.get_output(inputs)["pred._label"] # TYPE: pd
        acc = 0
        changed_count = 0
        confi_preds = []
        confi_golds = []

        print(preds)
        preds = text_acc(preds, is_generate)
        
        if is_ours:
            for idx in range(len(zero_shot_preds)):
                if zero_shot_standard[idx] == "Direct Answer":
                    changed_count += 1
                    confi_preds.append(zero_shot_preds[idx])
                    confi_golds.append(gold_labels[idx])
                    # preds[idx] = zero_shot_preds[idx]
            llm_zero_shot_acc = np.sum(np.where(np.array(confi_preds) == np.array(confi_golds), 1, 0)) *100 / len(confi_golds)
            print(f'changed {changed_count}')
            print(f'our confidence zero-shot LLM: {llm_zero_shot_acc}')   
      
        if task_name in ["chemprot", "acl_arc"]:
            f1_metric = evaluate.load("f1") 
            gold = np.array(gold_labels)      
            preds = np.array(preds)
            print(preds)

            f1_micro = f1_metric.compute(
                predictions=preds,
                references=gold,
                average="micro",
            )["f1"]

            f1_macro = f1_metric.compute(
                predictions=preds,
                references=gold,
                average="macro",
            )["f1"]

            if task_name == "chemprot":
                print(f"micro-F1 : {f1_micro*100:.2f}")
            else:
                print(f"macro-F1 : {f1_macro*100:.2f}")
            
        else:
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
    parser.add_argument('--upper_ratio', help='0~1', type=float, required=True)
    parser.add_argument('--back_ratio', help='0~1', type=float, required=True)
    # required = False
    parser.add_argument('--is_generate', help='generation vs classification', type=str, required=False, default="no")
    parser.add_argument('--is_seq2seq', help='using seq2seq', type=str, required=False, default="no")
    
    args = parser.parse_args()
    ratio = args.ratio
    back_ratio = args.back_ratio
    seed = args.seed
    target_task = args.target_task
    model_name = args.model_name
    if target_task == "piqa":
        num_labels = 2
        question_path = "./dataset/piqa/dev.jsonl"
        answer_path = "./dataset/piqa/dev-labels.lst"
        # ckpt = [1007,2015,3022,4030,5037,6042]
        ckpt = [4030]
        # ckpt = [1008*idx for idx in range(1,7)]
    elif target_task == "csqa":
        num_labels = 5
        question_path = "./dataset/csqa/dev_rand_split.jsonl"
        answer_path = "./dataset/csqa/dev_rand_split.jsonl"
        ckpt = [609*idx for idx in range(1,7)]
    elif target_task == "stqa":
        num_labels = 2
        question_path = "./dataset/stqa/dev.json"
        answer_path = "./dataset/stqa/dev.json"
        ckpt = [129*idx for idx in range(1,7)]
    elif target_task == "qasc":
        num_labels = 8
        question_path = "./dataset/qasc/dev.jsonl"
        answer_path = "./dataset/qasc/dev.jsonl"
        ckpt = [509*idx for idx in range(1,7)]
    elif target_task == "arc-e":
        num_labels = 4
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [141*idx for idx in range(1,7)]
    elif target_task =="arc-h":
        num_labels = 4
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        # ckpt = [373*idx for idx in range(1,7)] 
        ckpt = [559*idx for idx in range(2,7)] # batch size 2
    elif target_task =="obqa":
        num_labels = 4
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [310*idx for idx in range(2,7)]
    elif target_task =="wngr":
        num_labels = 2
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [2525*idx for idx in range(1,7)]
    elif target_task =="siqa":
        num_labels = 3
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        ckpt = [2089*idx for idx in range(1,7)]
    elif target_task =="boolq":
        num_labels = 2
        question_path = f"./dataset/{target_task}/dev.jsonl"
        answer_path = f"./dataset/{target_task}/dev.jsonl"
        # ckpt = [589, 1179, 1768, 2358, 2947, 3534]
        ckpt = [3540]
    elif target_task == "chemprot":
        num_labels = 13
        question_path = "./dataset/domain_classification/chemprot/test.txt"
        answer_path = None
        cq_path = f"./dataset/{target_task}/converted_train.json"
        ckpt = [261, 522, 783, 1044,1305,1566, 1827, 2088, 2349, 2610, 2871, 3132]
    elif target_task == "acl_arc":
        num_labels = 6
        question_path = "./dataset/domain_classification/citation_intent/test.txt"
        answer_path = None
        cq_path = f"./dataset/{target_task}/converted_train.json"
        ckpt = [105, 211, 316, 422, 527, 633, 738, 844,949,1055, 1160, 1260]
    
    
    
    if args.is_generate == "yes":
        is_generate = True
    else:
        is_generate = False
    
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
    
    if args.is_seq2seq == "yes":
        is_seq2seq = True
    else:
        is_seq2seq = False
    
    # epoch   
    for idx in ckpt:
        # current_model_name = f'./models/backbone_mistral/siqa/Llama-3.2-1B-Instruct/False/lora_True/42/0.0/checkpoint-{idx}'
        # current_model_name = f'./models/ablation/only_answer/{target_task}/cls/{model_name}/{is_ours}/{lora_True}/{seed}/{back_ratio}/checkpoint-{idx}'
        current_model_name = f'./models/backbone_13B/{target_task}/cls/{model_name}/{is_ours}/{lora_True}/{seed}/{back_ratio}/checkpoint-{idx}'
        print(f'current_ratio: {ratio}')    
        
 
        print(f'current model name is {current_model_name}')
        acc, preds_list, gold_labels = evaluater(current_model_name, target_task, is_ours, is_lora, is_generate, is_seq2seq, num_labels, ratio, back_ratio, args.upper_ratio)

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