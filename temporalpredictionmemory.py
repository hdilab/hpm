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
Temporal Prediction Memory class
Model a single temporal prediction layer
"""

import numpy as np
from cell import Cell, RandomModule
from collections import deque
import math

DEBUG = True # Print lots of information


class TemporalPredictionMemory(object):
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
                 cellsPerColumn=8,
                 dendritesPerCell=16,
                 synapsesPerDendrite=32,
                 activationThreshold=100,
                 connectedPermanence=0.5,
                 numColumn=512,
                 seed=42,
                 feeder=None,
                 numHistory=4,
                 updateWeight=0.2):
        self.feeder = feeder
        randomModule = RandomModule(seed=seed,
                                    numColumn=numColumn,
                                    cellsPerColumn=cellsPerColumn)

        self.columns = []
        self.predictedCells = np.full((numColumn, cellsPerColumn), False)
        self.activatedCells = np.full((numColumn, cellsPerColumn), False)
        self.energyCells = np.full((numColumn, cellsPerColumn), 1.0)
        self.historyActivatedCells = deque()
        self.numColumn = numColumn
        self.cellsPerColumn = cellsPerColumn
        self.updateWeight = updateWeight
        self.predictedColumns = np.full((numColumn), False)

        for i in range(numColumn):
            column = []
            for j in range(cellsPerColumn):
                cell = Cell(dendritesPerCell=dendritesPerCell,
                            randomModule=randomModule,
                            activationThreshold=activationThreshold,
                            connectedPermanence=connectedPermanence
                            )
                column.append(cell)
            self.columns.append(column)

        for i in range(numHistory):
            self.historyActivatedCells.append(np.full((numColumn, cellsPerColumn), False))



    def feedForward(self):
        inputChar, inputSDR = self.feeder.feed()
        # Print predicted output char
        self.feeder.evaluatePrediction(inputChar, self.predictedColumns)
        self.activate(inputSDR)
        self.update()
        self.predict()
        self.printStatistics()

    def printStatistics(self):
        num_activeSynapses = 0
        sum_activeSynapses = 0
        for column in self.columns:
            for cell in column:
                for d in cell.dendrites:
                    activeSynapses = d[d>0]
                    num_activeSynapses += activeSynapses.size
                    sum_activeSynapses += np.sum(activeSynapses)
        if DEBUG:
            print("Total Active Synapses: ", num_activeSynapses)
            print("Average Synaptic weight: ", sum_activeSynapses*1.0/num_activeSynapses)



    def activate(self, input):
        self.activatedCells = np.full(self.activatedCells.shape, False)
        numActivatedCells = 0
        numActivatedColumns = 0
        for columnIndex in input:
            tempActivated = False
            for cellIndex, cell in enumerate(self.columns[columnIndex]):
                if self.predictedCells[columnIndex, cellIndex]:
                    self.activatedCells[columnIndex, cellIndex] = True
                    tempActivated = True
                    numActivatedCells += 1
            if not tempActivated:
                self.burst(columnIndex)
            else:
                numActivatedColumns += 1

        if DEBUG:
            print("Total Number of Activated Cells: ", numActivatedCells)
            print("Total Number of Activated Columns: ", numActivatedColumns)

    def burst(self, columnIndex):
        for cellIndex, cell in enumerate(self.columns[columnIndex]):
            self.activatedCells[columnIndex, cellIndex] = True

    def predict(self):
        numPredictionCell = 0
        pv = []
        for columnIndex, column in enumerate(self.columns):
            for cellIndex, cell in enumerate(column):
                p = \
                    self.columns[columnIndex][cellIndex].getPredictionValue(self.activatedCells)
                pv.append(p)

        pv.sort(reverse=True)
        index = math.floor(len(pv) * 0.01)
        threshold = pv[index]

        for columnIndex, column in enumerate(self.columns):
            columnPrediction: bool = False
            for cellIndex, cell in enumerate(column):
                self.predictedCells[columnIndex,cellIndex] = \
                    self.columns[columnIndex][cellIndex].predictWithThreshold(self.activatedCells, threshold)
                if self.predictedCells[columnIndex, cellIndex]:
                    columnPrediction = True
                    numPredictionCell += 1
            self.predictedColumns[columnIndex] = columnPrediction
        if DEBUG:
            numPredictionColumn = 0
            for column in self.predictedColumns:
                if column:
                    numPredictionColumn += 1
            print("Number of Predicted cells: ", numPredictionCell)
            print("Number of Predicted columns: ", numPredictionColumn)




    def update(self):
        updateMask = self.historyActivatedCells[3] * self.updateWeight
        updateMask = updateMask + self.historyActivatedCells[2] * self.updateWeight * (-0.8)
        updateMask = updateMask + self.historyActivatedCells[1] * self.updateWeight * (-0.5)
        updateMask = updateMask + self.historyActivatedCells[0] * self.updateWeight * (-0.25)
        self.historyActivatedCells.popleft()
        self.historyActivatedCells.append(self.activatedCells)
        if len(updateMask[updateMask<0]) > 0 :
            uniques = np.unique(updateMask)
            uniques_count = {u:len(updateMask[updateMask==u]) for u in uniques}

            if DEBUG:
                for u in uniques:
                    print(u, uniques_count[u])
                print("Total update: ", sum(u*uniques_count[u] for u in uniques_count))

        for columnIndex, column in enumerate(self.columns):
            for cellIndex, cell in enumerate(column):
                cell.applyMask(updateMask)













