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


class memoAE(nn.Module):
    def __init__(self,
                 numBits=512,
                 name='memo AE',
                 numOnBits=10):
        super().__init__()

        learningRate = 1e-3
        self.name = name
        self.iteration = 0
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.encoder = nn.MaxPool1d(4).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(numBits, numBits*2),
            nn.LeakyReLU(),
            nn.Linear(numBits*2, numBits*4)).to(device)
        self.kWTARecon = Sparsify1D(numOnBits*4)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                     lr=learningRate,
                                     weight_decay=1e-5)
        self.loss = 0
        self.recall = 0
        self.targetSum = 0
        self.startTime = time.time()
        self.programStartTime = time.time()

    def forward(self, x):
        xDevice = torch.reshape(x, (-1,4)).to(device)
        out = xDevice[:,0]
        emb = torch.reshape(out, (1, -1)).to(device)
        recon = self.decoder(emb)
        return recon, emb

    def testBinaryEmbedding(self, x):
        recon, binaryEmb = self.forward(x)
        reconKWTA = self.kWTARecon(recon)
        binaryRecon = (reconKWTA != 0).to(recon).to(device)
        return binaryRecon, binaryEmb

    def pool(self, x, writer):
        xDevice = x.to(device)
        recon, binaryEmb = self.forward(xDevice)
        loss = self.criterion(recon, xDevice)
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
            recall, targetSum = self.getRecallError(input, binaryRecon)
            self.recall += recall
            self.targetSum += binaryEmb.sum()

        if self.iteration % printInterval == 0:
            self.loss /= printInterval
            self.recall /= printInterval
            self.targetSum /= printInterval

            endTime = time.time()
            trainingTime = int(endTime - self.startTime)
            totalTime = int((endTime - self.programStartTime)/3600)

            print(
                '{} [{}],  Train Loss:{:.6f}, Recall:{:.6f}, TargetSum:{:.2f} Training Time:{} Total Hour:{}'
                .format(self.name, self.iteration,  self.loss, self.recall, self.targetSum, trainingTime, totalTime))
            writer.add_scalar('loss/AE-BCE' + self.name, self.loss, self.iteration)
            writer.add_scalar('recall/AE-Recall' + self.name, self.recall, self.iteration)
            writer.add_scalar('sparcity/TargetSum' + self.name, self.targetSum, self.iteration)
            self.loss = 0
            self.recall = 0
            self.targetSum = 0
            self.startTime = endTime

    def getRecallError(self, target, pred):
        common = target * pred
        commonSum = common.sum()
        targetSum = target.sum()
        recall = commonSum / (targetSum + 0.0001)
        if recall > 0.99:
            print("Hello ", self.name)
        return recall, targetSum
