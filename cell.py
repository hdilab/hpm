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
        weights = np.random.rand(self.numColumn, self.cellsPerColumn)
        weights[column, :] = -1
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
                 connectedPermanence=0.5):
        self.activationThreshold = activationThreshold
        self.connectedPermanence = connectedPermanence
        self.dendrites = []
        self.column = column
        for d in range(dendritesPerCell):
            self.dendrites.append(randomModule.getRandomPermance(self.column))

    def predict(self, activatedCells):
        for d in self.dendrites:
            connectedSynapses = (d > self.connectedPermanence).astype(np.int)
            if np.sum(connectedSynapses * activatedCells) > self.activationThreshold:
                return True
        return False

    def predictWithThreshold(self, activatedCells, threshold):
        for d in self.dendrites:
            connectedSynapses = (d > self.connectedPermanence).astype(np.int)
            if np.sum(connectedSynapses * activatedCells) > threshold:
                return True
        return False

    def getPredictionValue(self, activatedCells):
        predictionValue = []
        for d in self.dendrites:
            connectedSynapses = (d > self.connectedPermanence).astype(np.int)
            pv = np.sum(connectedSynapses * activatedCells)
            predictionValue.append(pv)
        return max(predictionValue)

    def applyMask(self, mask):
        for i, d in enumerate(self.dendrites):
            activeSynapses = (d>0).astype(np.float)
            d = (d + mask) * activeSynapses
            d = np.minimum(d, np.ones(d.shape))
            self.dendrites[i] = d



