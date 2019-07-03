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
Text file Feeder class
Feeder feeds input to the Temporal Memory
This class feeds SDR from the input ASCII text file

Usage concept
txtfeeder = TXTFeeder("input.txt")
L1 = temporalmemory(feeder=txtfeeder)
L2feeder = TemporalFeeder(input=L1)
L2 = temporalmemory(feeder=L2feeder)
L2.feedforward()

def feedforward()
    input=feeder.feed()
    self.predict(input)
"""

from feeder import Feeder
from SDR import SDR
import numpy as np

class TXTFeeder(Feeder):
    """
      Class implementing the  Feeder.

      :param inputFileName: (str) Name of the input file

      :param numBits: (int) Number of bits for SDR. Default value ``512``

      :param numOnBits: (int) Number of Active bits for SDR. Default value ``10``.
            It is 2% sparcity for 512 bit

      :param seed: (int) Seed for the random number generator. Default value ``42``.
    """
    def __init__(self,
                 inputFileName,
                 numBits=512,
                 numOnBits=10,
                 seed=42,
                 ):
        Feeder.__init__(self, numBits, numOnBits)
        self.char_list = [char for char in open(inputFileName).read()]
        asc_chars = [chr(i) for i in range(128)]
        self.char_sdr = SDR(asc_chars,
                            numBits=numBits,
                            numOnBits=numOnBits,
                            seed=seed)
        self.readIndex = -1

    def feed(self):
        if self.readIndex < len(self.char_list)-1:
            self.readIndex = self.readIndex + 1
        else:
            self.readIndex = -1
        inputChar = self.char_list[self.readIndex]
        inputSDR = self.char_sdr.getSDR(inputChar)
        return inputChar, inputSDR

    def evaluatePrediction(self, inputChar, prediction):
        scores = [(i, self.getMatch(i, prediction)) for i in range(128)]
        scores = [s for s in scores if s[1] > 8]
        scores.sort(key=lambda x: x[1])
        print("Input: ", inputChar)
        for i, score in scores:
            print("\tPrediction:", chr(i), score)

    def getMatch(self, i, prediction):
        return np.sum(prediction[self.char_sdr.getSDR(chr(i))].astype(np.int))