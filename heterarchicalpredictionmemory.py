# ----------------------------------------------------------------------
# Heterachrical Prediction Memory
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
import pickle
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/exp-2', comment='Single layer, Non-overlapping text')

DEBUG = True # Print lots of information
PRINT_LOG = True # Will print the log of the accuracy

NUM_WEIGHT_DEC = -0.03
NUM_WEIGHT_INC = 0.3


class HeterarchicalPredictionMemory(object):
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
                 sizeSDR=512,
                 numOnBits=10,
                 seed=42,
                 lower=None,
                 dropbout=0.5):
        
        self.lower = lower
        np.random.seed(seed)
        self.weights = np.random.rand(sizeSDR, sizeSDR*2)
        self.iteration = 0
        self.prevActual = []
        self.dropout = dropbout
        self.numOnBits = numOnBits
        self.sizeSDR = sizeSDR
        self.prevDropout = np.zeros(self.weights.shape)
        self.accuracy = 0

    def printHeader(self):
        print("====================================")
        print("Iteration: ", self.iteration)
        print("====================================")

    def feed(self, context):
        output = []
        adjContext = [i+self.sizeSDR for i in context]
        for i in range(4):
            input = self.prevActual + adjContext
            self.iteration += 1
            if self.iteration == 4000:
                print("debug")
            pred = self.predict(input)
            char, actual = self.lower.feed(pred)
            self.evaluate(pred, actual)
            self.update(input, actual)
            self.prevActual = actual
            output.append(pred)
        output = self.pool(output)
        return output

    def predict(self, input):
        dropout = np.random.uniform(size=self.weights.shape)
        dropout = (dropout > self.dropout).astype(np.int)
        W = self.weights * dropout
        self.prevDropout = dropout
        dice = np.random.uniform(size=self.weights.shape)
        activation = (W > dice).astype(np.int)
        inputFull = np.zeros(self.sizeSDR*2)
        inputFull[input] = 1
        activation = activation * inputFull
        output = activation.sum(axis=1)
        result = np.sort(output)[::-1]
        result = result[:self.numOnBits]
        resultDict = {str(i):result[i] for i in range(self.numOnBits)}
        # writer.add_scalars('predict/predict_group', resultDict, self.iteration)
        writer.add_scalar('predict/predict_mean', np.mean(result), self.iteration)
        writer.add_scalar('predict/predict_std', np.std(result), self.iteration)
        output = (-output).argsort()[:self.numOnBits]
        return output

    def evaluate(self, pred, actual):
        intersection = [i for i in actual if i in pred]
        accuracy = len(intersection)*1.0/len(actual)
        self.accuracy = 0.99*self.accuracy + 0.01*accuracy
        if self.iteration %100 == 0:
            print("Iteration: \t", self.iteration, "Accuracy: \t", self.accuracy)
            writer.add_scalar('accuracy/acc', self.accuracy, self.iteration)
            writer.add_scalar('weights/mean', np.mean(self.weights), self.iteration)
            writer.add_scalar('weights/std', np.std(self.weights), self.iteration)
            writer.add_histogram('weights', self.weights, self.iteration)

    def update(self, input, actual):
        decMask = np.zeros(self.weights.shape)
        decMask[:,input] = 1
        incMask = np.zeros(self.weights.shape)
        incMask[actual, :] = 1
        incMask = decMask * incMask
        mask = decMask * NUM_WEIGHT_DEC + incMask * NUM_WEIGHT_INC
        mask = mask * self.prevDropout
        self.weights += mask

    def pool(self, output):
        # print(output)
        return output



def sparseToFull(sparse, size):
    output = np.zeros(size)
    output[sparse] = 1
    return output
