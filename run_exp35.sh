CUDA_VISIBLE_DEVICES=2 python exp35.py  -n R1L1-10k  -l 1 -i data/10k.txt &
CUDA_VISIBLE_DEVICES=3 python exp35.py  -n R1L2-10k  -l 2 -i data/10k.txt &
CUDA_VISIBLE_DEVICES=4 python exp35.py  -n R1L3-10k  -l 3 -i data/10k.txt &
