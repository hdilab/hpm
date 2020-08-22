from torch import nn
import torch
import numpy as np
from torch import unsqueeze
from models.NNSAE import NNSAE
from models.FC import FCML, FC

class HPM(object):
    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 lower=None,
                 name="layer"):
        super().__init__()
        # self.mlp = FC(numBits=512,
        #               numOnBits=10)
        self.mlp = FCML(inputDim=numBits,
                        hiddenDim=256,
                        outputDim=numBits)
        self.pooler = NNSAE( inputDim=numBits*4,
                             hiddenDim=numBits,
                             name=name+"-AE")
        self.lr = 0.00001
        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.prevActual = torch.rand((1, 1, numBits)) # vector holding current input
        self.prevActual[0] = 1
        self.actual = torch.rand((1, 1, numBits)) # vector holding current input
        self.pred = torch.rand((1, 1, numBits)) # vector holding current input

        self.printInterval = 1000
        self.losses = [ 0 for i in range(self.printInterval)]
        self.recalls = [0 for i in range(self.printInterval)]
        self.reconstructionErrors = [0 for i in range(self.printInterval)]
        self.reconstructionRecalls = [0 for i in range(self.printInterval)]
        self.iteration = 0

        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.MSELoss(reduction='sum')
    def feed(self, context):
        output = []
        for i in range(4):
            actual =  self.lower.feed(self.prevActual)
            self.actual = unsqueeze(torch.tensor(actual.T, dtype=torch.float32), 0)
            self.mlp.train()
            self.opt.zero_grad()
            self.pred = self.mlp(self.prevActual)
            loss = self.criterion(self.pred, self.actual)
            if self.iteration % self.printInterval == self.printInterval-1:
                print(self.iteration, loss.item())

            loss.backward()
            self.opt.step()


            with torch.no_grad():
                self.evaluate()
                self.prevActual = self.actual
                output.append(self.actual.squeeze().numpy())
                self.iteration += 1
        output = np.concatenate(output)
        output = self.pooler.pool(output)
        return output



    def evaluate(self):
        self.losses[self.iteration%self.printInterval] = \
            self.getMSE(self.actual, self.pred)
        self.recalls[self.iteration % self.printInterval] = \
            self.getRecallError(self.actual, self.pred)
        # reconstructPred = self.getReconstruction()
        # self.reconstructionErrors[self.iteration % self.printInterval] = \
        #     self.getMSE(self.prevActual, reconstructPred)
        # self.reconstructionRecalls[self.iteration % self.printInterval] = \
        #     self.getRecallError(self.prevActual, reconstructPred)


        if self.iteration % self.printInterval  == 0:
            accuracy = np.mean(self.losses)
            meanRecall = np.mean(self.recalls)
            meanReconstructionError = np.mean(self.reconstructionErrors)
            meanReconstructionRecall = np.mean(self.reconstructionRecalls)
            print(self.name, \
                  "\t Iteration: \t", self.iteration, \
                  "\t MSE: \t",  accuracy, \
                  "\t Recall: \t",  meanRecall, \
                  "\t Reconstruction MSE: \t", meanReconstructionError, \
                  "\t Reconstruction Recall: \t", meanReconstructionRecall)
            # writer.add_scalar('accuracy/loss', accuracy, self.iteration)

    def getRecallError(self, target, pred):
        targetSparse = np.asarray(target).flatten()
        targetSparse = np.where(targetSparse > 0.1)
        targetSparse = list(targetSparse[0])
        # targetSparse = list(targetSparse)
        numTarget = len(targetSparse)
        predSparse = np.asarray(pred).flatten()
        predSparse = np.argsort(predSparse)[-1 * numTarget:]
        predSparse = list(predSparse)
        intersection = [i for i in targetSparse if i in predSparse]
        recall = len(intersection) / (numTarget + 0.0001)
        return recall

    def getReconstruction(self):
        ind = np.argsort(self.h, axis=0)[-10:]
        reconstructH = np.zeros(self.h.shape)
        reconstructH[ind] = 1.0
        reconstructPred = self.W * reconstructH
        return reconstructPred

    def getMSE(self, target, pred):

        error = target.squeeze() - pred.squeeze()
        loss = (error.T @ error) / self.numBits
        return loss

