from torch import nn
import torch

class FC(nn.Module):
    def __init__(self,
                 numBits=512,
                 numOnBits=10):
        super().__init__()
        self.fc = nn.Linear(numBits, numBits)
        self.numOnBits = numOnBits

    def forward(self, x):
        out = self.fc(x)
        topval = out.topk(self.numOnBits, dim=1)[0][:,-1]
        comp = (out>=topval)
        return comp*out

