# # only background knowledge
# CUDA_VISIBLE_DEVICES=1 python fine_tune_lora_cls.py --model_ckpt meta-llama/Llama-3.2-1B-Instruct --task obqa --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task obqa --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9
# CUDA_VISIBLE_DEVICES=1 python fine_tune_lora_cls.py --model_ckpt meta-llama/Llama-3.2-1B-Instruct --task siqa --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task siqa --is_ours yes --is_lora yes --seed 42 --ratio 0.9 --back_ratio 0.9

