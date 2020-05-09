CUDA_VISIBLE_DEVICES=5  python exp15.py data/short.txt -n EXP-15-short -e 10000 -b 1 -s 10 &
CUDA_VISIBLE_DEVICES=6 python exp15.py data/medium.txt -n EXP-15-medium -e 10000 -b 10 -s 10 &
CUDA_VISIBLE_DEVICES=7 python exp15.py data/1342.txt -n EXP-15-long -e 10000 -b 128 -s 192 &



