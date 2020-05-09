CUDA_VISIBLE_DEVICES=5  python exp.py data/short.txt -n EXP-17-short -e 10000 -b 163840 &
CUDA_VISIBLE_DEVICES=8 python exp.py data/medium.txt -n EXP-17-medium -e 10000 -b 163840  &
CUDA_VISIBLE_DEVICES=9 python exp.py data/long.txt -n EXP-17-long -e 10000 -b 163840  &



