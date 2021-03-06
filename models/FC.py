from torch import nn
import torch

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
        # out = self.sigmoid(out)
        return out

class FCML(nn.Module):
    def __init__(self,
                 numBits=512,
                 numOnBits=10):
        super().__init__()
        self.fc1 = nn.Linear(numBits, numBits)
        self.fc2 = nn.Linear(numBits, numBits)
        self.sigmoid1 = nn.Sigmoid()
        # self.sigmoid2 = nn.Sigmoid()
        self.numOnBits = numOnBits

    def forward(self, x):
        out = self.sigmoid1(self.fc1(x))
        out = self.fc2(out)

        # out = self.sigmoid(out)
        return out