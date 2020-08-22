from torch import nn
import torch
import numpy as np
from torch import unsqueeze
from models.NNSAE import NNSAE

# writer = SummaryWriter('runs/exp-2', comment='Single layer, Non-overlapping text')

class FC_kWTA(nn.Module):
    def __init__(self,
                 numBits=512,
                 numOnBits=10):
        super().__init__()
        self.fc = nn.Linear(numBits, numBits)
        self.numOnBits = numOnBits

    def forward(self, x):
        out = self.fc(x)
        topval = out.topk(self.numOnBits, dim=1)[0][:,-1]
        topval = topval.repeat(out.shape[1], 1).permute(1,0).view_as(out)
        comp = (out>=topval).to(out)
        return comp


class FC(nn.Module):
    def __init__(self,
                 numBits=512,
                 numOnBits=10):
        super().__init__()
        self.fc = nn.Linear(numBits, numBits)
        self.sigmoid = nn.Sigmoid()
        self.numOnBits = numOnBits

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out

class FCML(nn.Module):
    def __init__(self,
                 inputDim=1024,
                 hiddenDim=256,
                 outputDim=512):
        super(FCML,self).__init__()
        self.fc1 = nn.Linear(inputDim, hiddenDim)
        self.fc2 = nn.Linear(hiddenDim, outputDim)

    def forward(self, x):
        a = self.fc1(x)
        h = torch.sigmoid(a)
        out = self.fc2(h)
        return out