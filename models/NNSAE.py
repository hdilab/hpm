# ----------------------------------------------------------------------
# Non-Negative Sparse AutoRegressive Model
# Copyright (C) 2019, HDILab.  Unless you have an agreement
# with HDILab, for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

"""
Heterarchical Prediction Memory class
Model a single temporal prediction layer
"""

import numpy as np
from tensorboardX import SummaryWriter
from numpy.matlib  import rand, zeros, ones

writer = SummaryWriter('runs/exp-2', comment='Single layer, Non-overlapping text')

class NNSAE(object):
    """
    Class implementing the Temporal Prediciton Memory.
    :param input_list: (List) List for input_values.
        For ASCII it will be [chr(0), chr(1), ... chr(127)]
    :param numBits: (int) Number of bits for SDR. Default value ``512``
    :param numOnBits: (int) Number of Active bits for SDR. Default value ``10``.
        It is 2% sparcity for 512 bit
    :param seed: (int) Seed for the random number generator. Default value ``42``.
    """

    def __init__(self,
                 inputDim=2048,
                 hiddenDim=512,
                 name="NNSAE"):

        self.inputDim = inputDim # number of input neurons
        self.hiddenDim = hiddenDim # number of hidden neurons

        self.inp = zeros((self.inputDim, 1)) # vector holding current input
        self.inp[0] = 1
        self.out = zeros((self.inputDim, 1)) # vector holding current input

        self.g = zeros((self.hiddenDim, 1)) # neural activity before non-linearity
        self.h = zeros((self.hiddenDim, 1)) # hidden neuron activation
        self.a = ones((self.hiddenDim, 1)) # slopes of activation functions
        self.b = -3*ones((self.hiddenDim, 1)) # biases of activation functions
        scale = 0.025

        # shared network weights, i.e. used to compute hidden layer activations and estimated outputs
        self.W = scale * (2 * rand((self.inputDim, self.hiddenDim)) -
                            0.5 * ones((self.inputDim, self.hiddenDim))) + scale

        # learning rate for synaptic plasticity of read-out layer (RO)
        self.lrateRO = 0.01
        self.regRO = 0.0002 # numerical regularization constant
        self.decayP = 0 # decay factor for positive weights [0..1]
        self.decayN = 1 # decay factor for negative weights [0..1]

        self.lrateIP = 0.001 # learning rate for intrinsic plasticity (IP)
        self.meanIP = 0.02 # desired mean activity, a parameter of IP
        self.name = name


        self.printInterval = 1000
        self.losses = [ 0 for i in range(self.printInterval)]
        self.recalls = [0 for i in range(self.printInterval)]
        self.reconstructionErrors = [0 for i in range(self.printInterval)]
        self.reconstructionRecalls = [0 for i in range(self.printInterval)]
        self.iteration = 0

    def pool(self, X):
        self.inp = np.expand_dims(X, axis=1)
        self.forward()
        self.evaluate()
        self.update()
        self.iteration += 1
        return self.h

        # Update network activation
        # This helper function computes the new activation pattern of the
        # hidden layer for a given input. Note that self.inp field has to be set in advance.

    def forward(self):
        # excite network
        self.g = self.W.T * self.inp
        # apply activation function
        self.h = 1 / (1 + np.exp(np.multiply(-self.a, self.g) - self.b))
        # read-out
        self.out = self.W * self.h

    def evaluate(self):
        self.losses[self.iteration % self.printInterval] = \
            self.getMSE(self.inp, self.out)
        self.recalls[self.iteration % self.printInterval] = \
            self.getRecallError(self.inp, self.out)
        reconstructPred = self.getReconstruction()
        self.reconstructionErrors[self.iteration % self.printInterval] = \
            self.getMSE(self.inp, reconstructPred)
        self.reconstructionRecalls[self.iteration % self.printInterval] = \
            self.getRecallError(self.inp, reconstructPred)

        if self.iteration % self.printInterval == 0:
            accuracy = np.mean(self.losses)
            meanRecall = np.mean(self.recalls)
            meanReconstructionError = np.mean(self.reconstructionErrors)
            meanReconstructionRecall = np.mean(self.reconstructionRecalls)
            print(self.name, \
                  "Iteration: \t", self.iteration, \
                  "\t MSE: \t", accuracy, \
                  "\t Recall: \t", meanRecall, \
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
        error = target - pred
        loss = (error.T * error) / self.inputDim
        return loss[0, 0]

    def update(self):
        # calculate adaptive learning rate
        lrate = self.lrateRO / (self.regRO + sum(np.power(self.h, 2)))

        # calculate error
        error = self.inp - self.out
        loss = (error.T * error) / self.inputDim

        # update weights
        self.W = self.W + lrate[0, 0] * (error * self.h.T)

        # decay function for positive weights
        if self.decayP > 0:
            idx = np.where(self.W > 0)
            self.W[idx] -= self.decayP * self.W[idx]

        # decay function for negative weights
        if self.decayN == 1:
            # pure NN weights!
            self.W = np.maximum(self.W, 0)
        else:
            if self.decayN > 0:
                idx = np.where(self.W < 0)
                self.W[idx] -= self.decayN * self.W[idx]

        # intrinsic plasticity
        hones = ones((self.hiddenDim, 1))
        tmp = self.lrateIP * \
              (hones - (2.0 + 1.0 / self.meanIP) *
               self.h + np.power(self.h, 2) / self.meanIP)
        self.b = self.b + tmp
        self.a = self.a + self.lrateIP * hones / \
                 self.a + np.multiply(self.g, tmp)
