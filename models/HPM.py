from torch import nn
import torch
import numpy as np
from models.AE import kWTA_autoencoder
from models.FC import FCML, FC
import time

class HPM(object):
    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 lower=None,
                 name="layer"):

        super().__init__()
        # self.mlp = FC(inputDim=512,
        #               outputDim=512)
        # self.net = FCML(inputDim=numBits * 2,
        #                 hiddenDim=256,
        #                 outputDim=numBits)
        self.net = FCML(inputDim=numBits * 2,
                        hiddenDim=256,
                        outputDim=numBits)
        # self.pooler = NNSAE( inputDim=numBits*4,
        #                      hiddenDim=numBits,
        #                      name=name+"-AE")
        self.pooler = kWTA_autoencoder(numBits=numBits,
                                       name=name+"-AE")
        self.lr = 0.0001
        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.prevActual = self.getSparseBinary(torch.rand((1, numBits)), numOnBits) # vector holding current input
        self.actual = self.getSparseBinary(torch.rand((1, numBits)), numOnBits) # vector holding current input
        self.pred = self.getSparseBinary(torch.rand((1, numBits)), numOnBits) # vector holding current input


        self.printInterval = 5000
        self.losses = [ 0 for i in range(self.printInterval)]
        self.recalls = [0 for i in range(self.printInterval)]
        self.reconstructionErrors = [0 for i in range(self.printInterval)]
        self.reconstructionRecalls = [0 for i in range(self.printInterval)]
        self.bceloss = [0 for i in range(self.printInterval)]
        self.iteration = 0

        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()
        self.startTime = time.time()
        # self.criterion = nn.MSELoss(reduction='sum')
    def feed(self, context = None, writer=None):
        output = []
        for i in range(4):
            actual =  self.lower.feed(context=self.pred, writer=writer)
            # self.actual = unsqueeze(torch.tensor(actual.T, dtype=torch.float32), 0)
            self.actual = torch.tensor(actual, dtype=torch.float32)
            # self.mlp.train()
            # self.net.zero_grad()
            self.opt.zero_grad()
            input = torch.cat((self.prevActual, context), dim=1)
            self.pred = self.net(input)
            loss = self.criterion(self.pred, self.actual)
            # if self.iteration % self.printInterval == self.printInterval-1:
            #     print(self.iteration, loss.item())
            self.bceloss[self.iteration % self.printInterval] = loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            self.opt.step()

            self.pred = self.pred.detach()
            self.pred = self.getSparseBinary(self.pred, self.numOnBits)
            self.evaluate(writer)

            self.prevActual = self.actual.detach()
            output.append(self.actual.squeeze().numpy())
            self.iteration += 1
        output = np.concatenate(output)
        pooler_input = torch.tensor(output)
        pooler_input = torch.reshape(pooler_input, (1,-1))
        pooler_output = self.pooler.pool(pooler_input, writer)
        return pooler_output.numpy()



    def evaluate(self,writer):
        actual = self.actual.detach().numpy()
        pred = self.pred.detach().numpy()
        # self.losses[self.iteration%self.printInterval] = \
        #     self.getMSE(actual, pred)
        self.recalls[self.iteration % self.printInterval] = \
            self.getRecallError(actual, pred)
        # reconstructPred = self.getReconstruction()
        # self.reconstructionErrors[self.iteration % self.printInterval] = \
        #     self.getMSE(self.prevActual, reconstructPred)
        # self.reconstructionRecalls[self.iteration % self.printInterval] = \
        #     self.getRecallError(self.prevActual, reconstructPred)


        if self.iteration % self.printInterval  == 0:
            # accuracy = np.mean(self.losses)
            meanRecall = np.mean(self.recalls)
            bce = np.mean(self.bceloss)
            currentTestTime = time.time()
            trainTime = int(currentTestTime - self.startTime)
            self.startTime = currentTestTime

            print(self.name, \
                  "\t Iteration: \t", self.iteration, \
                  "\t BCELoss: \t", bce, \
                  # "\t MSE: \t",  accuracy, \
                  "\t Recall: \t",  meanRecall,
                  "\t Training Time: \t", trainTime)
            writer.add_scalar('loss/BCE'+self.name, bce, self.iteration)
            writer.add_scalar('recall/recall'+self.name, meanRecall, self.iteration)


    def getRecallError(self, target, pred):
        targetSparse = target[0]
        targetIdx = np.where(targetSparse > 0.1)
        targetIdx = list(targetIdx[0])
        # targetSparse = list(targetSparse)
        numTarget = len(targetIdx)
        predSparse = pred[0]
        predSparse = np.argsort(predSparse)[-1 * numTarget:]
        predSparse = list(predSparse)
        intersection = [i for i in targetIdx if i in predSparse]
        recall = len(intersection) / (numTarget + 0.0001)
        # if recall > 0.99:
        #     print("Hello ", self.name)
        return recall

    def getSortedIndex(self, target):
        targetSparse = target[0]
        targetIdx = np.where(targetSparse > 0.1)
        sorted = np.sort(targetIdx)
        return sorted

    def getReconstruction(self):
        ind = np.argsort(self.h, axis=0)[-10:]
        reconstructH = np.zeros(self.h.shape)
        reconstructH[ind] = 1.0
        reconstructPred = self.W * reconstructH
        return reconstructPred

    def getMSE(self, target, pred):
        error = target - pred
        loss = (error.T @ error) / self.numBits
        return loss

    def getSparseBinary(self, dense, k):
        topVal = dense.topk(k)[0][:,-1]
        sparseBinary = (dense>=topVal).to(dense)
        return sparseBinary


