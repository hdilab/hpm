# Simple autoencoder
# Source code from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
import time

from torch import nn
from models.kWTA import Sparsify1D
import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
printInterval = 5000

class kWTA_autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='kWTA autoencoder',
                 numOnBits=10):
        super().__init__()

        learningRate = 1e-3
        self.name = name
        self.iteration = 0
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.encoder = nn.Sequential(
            nn.Linear(numBits*4, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits)).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(numBits, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits*4)).to(device)
        self.kWTA = Sparsify1D(numOnBits)
        self.kWTARecon = Sparsify1D(numOnBits*4)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                     lr=learningRate,
                                     weight_decay=1e-5)
        self.loss = 0
        self.recall = 0
        self.startTime = time.time()

    def forward(self, x):
        xDevice = x.to(device)
        out = self.encoder(xDevice)
        emb = self.kWTA(out)
        recon = self.decoder(emb)
        return recon, emb

    def testBinaryEmbedding(self, x):
        xDevice = x.to(device)
        out = self.encoder(xDevice)
        emb = self.kWTA(out)
        binaryEmb = (emb != 0).to(emb).to(device)
        recon = self.decoder(binaryEmb)
        reconKWTA = self.kWTARecon(recon)
        binaryRecon = (reconKWTA !=0).to(recon).to(device)  
        return binaryRecon, binaryEmb

    def pool(self, x, writer):
        xDevice = x.to(device)
        recon, emb = self.forward(xDevice)
        binaryEmb = (emb != 0).to(emb)
        loss = self.criterion(recon, xDevice) + torch.mean((binaryEmb - emb)*(binaryEmb-emb))
        self.loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.evaluate(xDevice.detach(), writer)
        # self.update()
        self.iteration += 1
        return binaryEmb

    def evaluate(self, input, writer):
        with torch.no_grad():
            binaryRecon, binaryEmb = self.testBinaryEmbedding(input)
            self.recall += self.getRecallError(input, binaryRecon)

        if self.iteration % printInterval == 0:
            self.loss /= printInterval
            self.recall /= printInterval

            endTime = time.time()
            trainingTime = int(endTime - self.startTime)

            print(
                '{} [{}],  Train Loss:{:.6f}, Recall:{:.6f},  Training Time:{} '
                .format(self.name, self.iteration,  self.loss, self.recall, trainingTime))
            writer.add_scalar('train/AE-BCE' + self.name, self.loss, self.iteration)
            writer.add_scalar('test/AE-Recall' + self.name, self.recall, self.iteration)
            self.loss = 0
            self.recall = 0
            self.startTime = endTime

    def getRecallError(self, target, pred):
        common = target * pred
        commonSum = common.sum()
        targetSum = target.sum()
        recall = commonSum / (targetSum + 0.0001)
        # if recall > 0.99:
        #     print("Hello ", self.name)
        return recall
