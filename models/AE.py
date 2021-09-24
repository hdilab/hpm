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
        self.encoderL1 = nn.Linear(numBits*4, numBits*2)
        self.encoderL2 = nn.Linear(numBits*2, numBits)
        self.decoderL1 = nn.Linear(numBits, numBits*2)
        self.decoderL2 = nn.Linear(numBits*2, numBits*4)

    def forward(self, x):
        k = self.numOnBits
        L1out = self.encoderL1(x);
        tmpL1 = L1out.view(x.shape[0],-1)
        L1TopVal = tmpL1.topk(k * 2)[0][:, -1]
        L1Sparse = (L1out > L1TopVal).to(L1out)

        L2out = self.encoderL2(L1Sparse)
        L2TopVal = L2out.topk(k * 1)[0][:, -1]
        sparseEmb = (L2out > L2TopVal).to(L2out)

        decoderL1out = self.decoderL1(sparseEmb);
        decoderL1TopVal = decoderL1out.topk(k * 2)[0][:, -1]
        decoderL1Sparse = (decoderL1out > decoderL1TopVal).to(decoderL1out)

        decoderL2out = self.decoderL2(decoderL1Sparse)
        decoderL2TopVal = decoderL2out.topk(k * 4)[0][:, -1]
        out = (decoderL2out > decoderL2TopVal).to(decoderL2out)
        return out, sparseEmb

