CUDA_VISIBLE_DEVICES=5  python exp.py data/short.txt -n EXP-16-short -e 10000 -b 64 &
CUDA_VISIBLE_DEVICES=1 python exp.py data/medium.txt -n EXP-16-medium -e 10000 -b 256  &
CUDA_VISIBLE_DEVICES=7 python exp.py data/long.txt -n EXP-16-long -e 10000 -b 256  &



