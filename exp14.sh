CUDA_VISIBLE_DEVICES=1  python exp14.py data/short.txt -n EXP-14-short -e 10000 -b 1 -s 10 &
CUDA_VISIBLE_DEVICES=2 python exp14.py data/medium.txt -n EXP-14-medium -e 10000 -b 10 -s 10 &
CUDA_VISIBLE_DEVICES=4 python exp14.py data/1342.txt -n EXP-14-long -e 10000 -b 128 -s 192 &



