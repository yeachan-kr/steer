# # only background knowledge

CUDA_VISIBLE_DEVICES=0 python eval.py --model_name Llama-3.2-1B-Instruct --target_task obqa --is_ours yes --is_lora yes --seed 42 --ratio 0.7 --back_ratio 0.9
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task arc-h --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task stqa --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task qasc --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task csqa --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9
# CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task piqa --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9