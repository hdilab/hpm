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
                 updateWeight=0.1):
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
        self.iteration = 0

        for i in range(numColumn):
            column = []
            for j in range(cellsPerColumn):
                cell = Cell(
                            i,
                            dendritesPerCell=dendritesPerCell,
                            randomModule=randomModule,
                            activationThreshold=activationThreshold,
                            connectedPermanence=connectedPermanence
                            )
                column.append(cell)
            self.columns.append(column)


        # for i in range(numHistory):
        #     self.historyActivatedCells.append(np.full((numColumn, cellsPerColumn), False))
            self.historyActivatedCells = np.full((numColumn, cellsPerColumn), False)



    def feedForward(self):
        print("====================================")
        print("Iteration: ", self.iteration)
        print("====================================")
        self.iteration += 1
        inputChar, inputSDR = self.feeder.feed()
        # Print predicted output char
        self.feeder.evaluatePrediction(inputChar, self.predictedColumns)
        self.activate(inputSDR)
        self.predict()
        self.update()
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
        numPredictedActivatedCells = 0
        numPredictedActivatedColumns = 0
        numBurstCells = 0
        numBurstColumns = 0
        for columnIndex in input:
            tempActivated = False
            for cellIndex, cell in enumerate(self.columns[columnIndex]):
                if self.predictedCells[columnIndex, cellIndex]:
                    self.activatedCells[columnIndex, cellIndex] = True
                    tempActivated = True
                    numPredictedActivatedCells += 1
            if not tempActivated:
                self.burst(columnIndex)
                numBurstCells += len(self.columns[columnIndex])
                numBurstColumns += 1
            else:
                numPredictedActivatedColumns += 1

        if DEBUG:
            print("Total Number of Predicted and Activated Cells: ", numPredictedActivatedCells)
            print("Total Number of Predicted and Activated Columns: ", numPredictedActivatedColumns)

            print("Total Number of Burst Cells: ", numBurstCells)
            print("Total Number of Burst Columns: ", numBurstColumns)

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
        index = math.floor(len(pv) * 0.04)
        threshold = pv[index]

        for columnIndex, column in enumerate(self.columns):
            columnPrediction: bool = False
            for cellIndex, cell in enumerate(column):
                maskedActivatedCells = np.copy( self.activatedCells )

                for ci in range(len(column)):
                    maskedActivatedCells[columnIndex, ci] = False

                self.predictedCells[columnIndex,cellIndex] = \
                    self.columns[columnIndex][cellIndex].predictWithThreshold(maskedActivatedCells, threshold)
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

            if numPredictionCell == 0:
                print("No cell predicted")

    def update(self):
        for columnIndex, column in enumerate(self.columns):
            for cellIndex, cell in enumerate(column):
                if self.activatedCells[columnIndex, cellIndex]:
                    # Currently activated cell case
                    # increase the synapses that connect previously activated cells and currently activated cells
                    updateMask = self.historyActivatedCells * self.updateWeight
                else:
                    # Currently not activated cell case
                    # decrease the synapses that connect previously activated cells and currently activated cells
                    updateMask = self.historyActivatedCells * self.updateWeight * (-0.01)

                cell.applyMask(updateMask)
                self.historyActivatedCells = self.activatedCells












