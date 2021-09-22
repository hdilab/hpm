# Simple autoencoder
# Source code from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

from torch import nn


class autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='SimpleAE'):
        super().__init__()

        self.numBits = numBits
        self.encoder = nn.Sequential(
            nn.Linear(numBits*4, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits))
        self.decoder = nn.Sequential(
            nn.Linear(numBits, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits*4))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

