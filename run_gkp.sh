for item in obqa wngr siqa piqa csqa qasc arc-h
  do
    CUDA_VISIBLE_DEVICES=0 python zero_shot_white_gkp.py --task $item --baseline selftalk
  done