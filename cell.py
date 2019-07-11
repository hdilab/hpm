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
Cell class
Handles issues with Cell

RandomModule is a helper class to generate random synapse with
"""

import numpy as np

class RandomModule(object):
    def __init__(self,
                 seed=42,
                 numColumn=512,
                 cellsPerColumn=32):
        np.random.seed(seed)
        self.numColumn = numColumn
        self.cellsPerColumn = cellsPerColumn

    def getRandomPermance(self, column):
        """
        We use gaussian distribution to get synapse weight
        If it is less than 0 it means they are permanently not connected.
        About half of the cells will be permanently disconnected
        We use standard deviation to be 0.15. The reason is when we
        calculate how many cells are connected from the beginning
        the following fomula gives us about 20, which looks reasonable
        np.sum(np.random.randn(2042,32)*0.15 > 0.5)
        stdPermanence is the design parameter we need to tune
        Finally we remove the connection that is within same column
        :param column:
        :return:
        """
        stdPermanence = 0.15
        weights = np.random.randn(self.numColumn, self.cellsPerColumn) * stdPermanence
        weights[column, :] = -1
        weights = np.minimum(weights, 1)
        return weights


class Cell(object):
    """
      Class implementing the Cell.

      :param input_list: (List) List for input_values.
            For ASCII it will be [chr(0), chr(1), ... chr(127)]

      :param numBits: (int) Number of bits for SDR. Default value ``512``

      :param numOnBits: (int) Number of Active bits for SDR. Default value ``10``.
            It is 2% sparcity for 512 bit

      :param seed: (int) Seed for the random number generator. Default value ``42``.
    """

    def __init__(self,
                 column,
                 dendritesPerCell=2,
                 randomModule=None,
                 activationThreshold=5,
                 connectedPermanence=0.5,
                 updateWeight=0.1):
        self.activationThreshold = activationThreshold
        self.connectedPermanence = connectedPermanence
        self.dendrites = []
        self.column = column
        self.updateWeight=updateWeight
        for d in range(dendritesPerCell):
            self.dendrites.append(randomModule.getRandomPermance(self.column))

    def predict(self, activatedCells):
        for d in self.dendrites:
            if self.predictDendrite(d, activatedCells):
                return True
        return False

    def predictDendrite(self, d, activatedCells):
        connectedSynapses = (d > self.connectedPermanence).astype(np.int)
        if np.sum(connectedSynapses * activatedCells) > self.activationThreshold:
            return True
        else:
            return False

    def getDendritePredictionValue(self, d, activatedCells):
        connectedSynapses = (d > self.connectedPermanence).astype(np.int)
        return np.sum(connectedSynapses * activatedCells)


    def getPredictionValue(self, activatedCells):
        predictionValue = []
        for d in self.dendrites:
            predictionValue.append(self.getDendritePredictionValue(d, activatedCells))
        return max(predictionValue)

    def strenthenActivatedDendrites(self, activatedCells):
        for d in self.dendrites:
            if self.predictDendrite(d, activatedCells):
                self.strengthenDendrite(d)

    def strengthenDendrite(self, d, activatedCells):
        enabledSynapses = (d > 0)
        d += enabledSynapses * activatedCells * self.updateWeight \
             - enabledSynapses * np.invert(activatedCells) * self.updateWeight * 0.1
        d = np.minimum(d, 1)

    def weakenActivatedDendrites(self, activatedCells):
        for d in self.dendrites:
            if self.predictDendrite(d, activatedCells):
                d -= activatedCells * self.updateWeight * 0.01

    def strengthenMaximumDendrite(self, activatedCells):
        prediction = []
        for d in self.dendrites:
            prediction.append((d, self.getDendritePredictionValue(d, activatedCells)))
        prediction.sort(key=lambda i:i[1], reverse=True)
        self.strengthenDendrite(prediction[0][0],activatedCells)







