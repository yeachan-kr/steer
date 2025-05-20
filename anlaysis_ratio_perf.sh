for item in stqa arc-h qasc csqa obqa piqa
do

  CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Llama-3.2-1B-Instruct --target_task $item --is_ours no --is_lora yes --seed 42 --ratio 0 --back_ratio 0.9
done
