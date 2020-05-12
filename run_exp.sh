CUDA_VISIBLE_DEVICES=2  python exp.py data/100.txt -n EXP-18-100 -e 10000 -b 10 &
CUDA_VISIBLE_DEVICES=3 python exp.py data/1k.txt -n EXP-18-1k -e 10000 -b 10  &
CUDA_VISIBLE_DEVICES=4 python exp.py data/10k.txt -n EXP-18-10k -e 10000 -b 64  &



