for item in arc-h obqa wngr siqa piqa csqa qasc  
  do
    CUDA_VISIBLE_DEVICES=1 python zero_shot_white_cot.py --task $item --baseline zs_ps_plus
  done