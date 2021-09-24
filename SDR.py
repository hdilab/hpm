# ----------------------------------------------------------------------
# Hierachrical Prediction Memory
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
SDR class
Handles issues with SDR
Given a char input, generate SDR
"""

import random
import numpy as np
from numpy import zeros

class SDR(object):
    """
      Class implementing the SDR.

      :param input_list: (List) List for input_values.
            For ASCII it will be [chr(0), chr(1), ... chr(127)]

      :param numBits: (int) Number of bits for SDR. Default value ``512``

      :param numOnBits: (int) Number of Active bits for SDR. Default value ``10``.
            It is 2% sparcity for 512 bit

      :param seed: (int) Seed for the random number generator. Default value ``42``.
    """

    def __init__(self,
                 input_list,
                 numBits=512,
                 numOnBits=10,
                 inputNoise=0.1,
                 seed=42):
        self.population = [i for i in range(numBits)]
        random.seed(seed)
        self.numOnBits = numOnBits
        self.numBits = numBits
        self.inputNoise = inputNoise
        self.sdr_dict = {i:random.sample(self.population, numOnBits) for i in input_list}
        self.inputList = input_list


    def getSDR(self, input):
        return self.sdr_dict[input]


    def getNoisySDR(self, input):
        inputSDR = self.sdr_dict[input]
        inputSDR = [i for i in inputSDR if random.random() > self.inputNoise]
        noise = random.sample(self.population, int(self.numOnBits * self.inputNoise))
        return inputSDR + noise

    def getDenseFromSparse(self, sparseInput):
        dense = zeros(self.numBits)
        dense[sparseInput] = 1
        dense = dense.reshape(-1,1)
        return dense

    def getSparseFromDense(self, denseInput):
        sparse = ( denseInput > 0 ).nonzero()
        return sparse



    def getInput(self, denseSDR):
        """
        Need to implement the function which returns the corresponding input from SDR
        This requires a probabilistic approach. Count the number of overlapping bit and nonoverlapping field.
        """
        count = {i:0 for i in self.inputList }
        sparseSDR = self.getSparseFromDense(denseSDR)[0]
        for b in sparseSDR:
            for c in self.inputList:
                if b in self.sdr_dict[c]:
                    count[c] += 1
        prediction = max(count, key=count.get)
        return prediction

    def getCollisionProb(self, n, a, s, theta):
        """
        Calculating the probability for the cases where more than theta synapses are activated
        for different cell activation pattern
        :param n: Number of cells
        :param a: Number of active cells
        :param s: Number of synapses
        :param theta: Threshold for the dendritic activation
        :return: The probability where dendritic activation for the different cell activation pattern
        """
        numerator = 0
        for b in range(theta, s+1):
            numerator += combinatorial(s, b) * combinatorial(n-s, a-b)

        denominator = combinatorial(n, a)

        return numerator*1.0/denominator

    def getRandomSDR(self):
        noise = random.sample(self.population, numOnBits)
        return noise


def combinatorial(a,b):
    return factorial(a)*1.0/factorial(a-b)/factorial(a)

def factorial(a):
    if a == 1:
        return 1
    else:
        return a*factorial(a-1)

