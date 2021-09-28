import torch.nn as nn


class Sparsify1D(nn.Module):
    def __init__(self,
                 numOnBits = 10):
        super().__init__()
        self.k = numOnBits

    def forward(self, x):
        tmpx = x.view(x.shape[0],-1)
        topval = tmpx.topk(self.k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        comp = (x>=topval).to(x)
        return comp*x