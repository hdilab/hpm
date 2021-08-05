# HTM
Heterarchical Prediction Memory (HPM)


## Environment
- python 2.7


## Experiment #21 
- Used NNASE for sparse autoencoder
- To replicate 

```shell
git checkout EXP-21
conda activate hpm
python exp21.py data/old_medium.txt -n EXP-21-Old-medium -e 100000
tensorboard --logdir runs --bind_all
```


![](docs/figures/exp21.png)

