from torch import nn
import torch

class FC(nn.Module):
    def __init__(self,
                 numBits=512):
        super().__init__()
        self.fc = nn.Linear(numBits, numBits)

    def forward(self, x):
        out = self.fc(x)
        return out

