CUDA_VISIBLE_DEVICES=2 python exp24.py data/old_medium.txt -n EXP-24-L1 -e 10000000 -l 1 &
CUDA_VISIBLE_DEVICES=3 python exp24.py data/old_medium.txt -n EXP-24-L2 -e 10000000 -l 2 &
CUDA_VISIBLE_DEVICES=4 python exp24.py data/old_medium.txt -n EXP-24-L3 -e 10000000 -l 3 &
CUDA_VISIBLE_DEVICES=5 python exp24.py data/old_medium.txt -n EXP-24-L4 -e 10000000 -l 4 &
