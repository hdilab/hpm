# Simple autoencoder
# Source code from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

from torch import nn
from models.kWTA import Sparsify1D
import torch

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

class simple_autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='simple_AE',
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
        x = self.decoder(emb)
        return x, emb

class simple_autoencoder2(nn.Module):

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
        L1out = self.encoderL1(x);
        L2input = nn.functional.relu(L1out)
        L2out = self.encoderL2(L2input)

        decoderL1out = self.decoderL1(L2out);
        decoderL2input = nn.functional.relu(decoderL1out)
        decoderL2out = self.decoderL2(decoderL2input)
        return decoderL2out, L2out


class kWTA_autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='kWTA autoencoder',
                 numOnBits=10):
        super().__init__()

        learningRate = 1e-3
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
        self.kWTA = Sparsify1D(numOnBits)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                     lr=learningRate,
                                     weight_decay=1e-5)

    def forward(self, x):
        out = self.encoder(x)
        emb = self.kWTA(out)
        x = self.decoder(emb)
        return x, emb

    def testBinaryEmbedding(self, x):
        out = self.encoder(x)
        emb = self.kWTA(out)
        binaryEmb = (emb != 0).to(emb)
        x = self.decoder(binaryEmb)
        return x, binaryEmb

    def pool(self, x, writer):
        recon, emb = self.forward(x)
        binaryEmb = (emb != 0).to(emb)
        loss = self.criterion(recon, x) + torch.sum(torch.abs(binaryEmb - emb))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.evaluate(writer)
        # self.update()
        # self.iteration += 1
        return binaryEmb