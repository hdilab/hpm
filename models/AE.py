# Simple autoencoder
# Source code from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

from torch import nn


class autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='kWTA_AE',
                 numOnBits=10):
        super().__init__()

        self.numBits = numBits
        self.numOnBits = numOnBits
        self.encoder = nn.Sequential(
            nn.Linear(numBits*4, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits))
        self.decoder = nn.Sequential(
            nn.Linear(numBits, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits*4))

    def forward(self, x):
        emb = self.encoder(x)
        k = self.numOnBits
        tmpx = emb.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(emb)
        comp = (emb >= topval).to(emb)
        sparseEmb = comp
        x = self.decoder(sparseEmb)
        return x, sparseEmb

