CUDA_VISIBLE_DEVICES=5  python exp.py data/100.txt -n EXP-17-100 -e 10000 -b 10 &
CUDA_VISIBLE_DEVICES=8 python exp.py data/1k.txt -n EXP-17-1k -e 10000 -b 10  &
CUDA_VISIBLE_DEVICES=9 python exp.py data/10k.txt -n EXP-17-10k -e 10000 -b 64  &



