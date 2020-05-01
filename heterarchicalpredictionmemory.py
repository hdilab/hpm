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
import random
import pickle
import time;
localtime = time.asctime( time.localtime(time.time()) )
from tensorboardX import SummaryWriter
writer = SummaryWriter('../runs/exp-11-' + localtime, comment='Sparse version Three dimensional Matrix')

DEBUG = True # Print lots of information
PRINT_LOG = True # Will print the log of the accuracy

NUM_WEIGHT_DEC = -0.05
NUM_WEIGHT_INC = 0.4


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
                 dropbout=0.5,
                 name="layer"):
        
        self.lower = lower
        np.random.seed(seed)
        self.weights = {}
        self.iteration = 0
        self.sizeSDR = sizeSDR
        self.numOnBits = numOnBits
        self.population = [i for i in range(self.sizeSDR)]
        self.prevActual = random.sample(self.population, self.numOnBits)
        self.dropout = dropbout

        self.accuracy = 0
        self.name=name

    def printHeader(self):
        print("====================================")
        print("Iteration: ", self.iteration)
        print("====================================")

    def feed(self, context):
        output = []

        for i in range(4):

            self.iteration += 1
            if self.iteration == 4000:
                print("debug")
            pred = self.predict(self.prevActual,context)
            actual = self.lower.feed(pred)
            self.evaluate(pred, actual)
            self.update(self.prevActual, context, actual)
            self.prevActual = actual
            output.append(actual)
        output = self.pool(output)
        return output

    def predict(self, input,context):
        activation = {}
        for o in self.weights:
            for i in input:
                for c in context:
                    if (i,c) in self.weights[o]:
                        if random.random() < self.weights[o][(i,c)]['value']:
                            if o in activation:
                                activation[o] += 1
                            else:
                                activation[o] = 1
        sortedActivation = sorted(activation.items(),key=lambda kv: kv[1], reverse=True)
        sortedActivation = sortedActivation[:self.numOnBits]
        output = [i[0] for i in sortedActivation]
        result = [i[1] for i in sortedActivation]

        if len(output) < self.numOnBits:
            output +=   random.sample(self.population, self.numOnBits - len(output))

        # writer.add_scalar('predict/predict_mean' + self.name, np.mean(result), self.iteration)
        # writer.add_scalar('predict/predict_std'+ self.name, np.std(result), self.iteration)

        return output

    def evaluate(self, pred, actual):
        intersection = [i for i in actual if i in pred]
        if len(actual) == 0:
            accuracy = 0
        else:
            accuracy = len(intersection)*1.0/len(actual)
        self.accuracy = 0.99*self.accuracy + 0.01*accuracy
        if self.iteration %100 == 0:
            print("Iteration: \t", self.iteration, "Accuracy: \t", self.accuracy)

            writer.add_scalar('accuracy/numSDR'+ self.name, len(actual), self.iteration)
            writer.add_scalar('accuracy/acc'+ self.name, self.accuracy, self.iteration)
            # writer.add_scalar('weights/mean'+ self.name, np.mean(self.weights), self.iteration)
            # writer.add_scalar('weights/std' + self.name, np.std(self.weights), self.iteration)
            # writer.add_histogram(self.name + 'weights', self.weights, self.iteration)

    def update(self, input, context, actual):
        for p in self.population:
            if p in actual:
                if p not in self.weights:
                    self.weights[p] = {}
                for i in input:
                    for c in context:
                        if (i,c) in self.weights[p]:
                            self.weights[p][(i,c)]['value'] += NUM_WEIGHT_INC
                        else:
                            self.weights[p][(i, c)] = {'value':NUM_WEIGHT_INC}
            else:
                if p in self.weights:
                    for i in input:
                        for c in context:
                            if (i,c) in self.weights[p]:
                                self.weights[p][(i,c)]['value'] += NUM_WEIGHT_DEC
                                if self.weights[p][(i,c)]['value'] < 0:
                                    del self.weights[p][(i,c)]


    def pool(self, output):
        # print(output)
        pooled = []
        for i in range(4):
            a = [int(e/4 + i*self.sizeSDR/4) for e in output[i] if e%4==0]
            pooled += a
        writer.add_scalar('pool/num' + self.name, len(pooled), self.iteration)
        # if len(pooled) < self.numOnBits:
        #     pooled +=   random.sample(self.population, self.numOnBits - len(pooled))
        return pooled



def sparseToFull(sparse, size):
    output = np.zeros(size)
    output[sparse] = 1
    return output
