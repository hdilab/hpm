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
import torch

# DEBUG = True # Will print lots of information
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))

class TXTFeeder(Feeder):
    """
      Class implementing the  Feeder.

      :param inputFileName: (str) Name of the input file

      :param numBits: (int) Number of bits for SDR. Default value ``512``

      :param numOnBits: (int) Number of Active bits for SDR. Default value ``10``.
            It is 2% sparcity for 512 bit

      :param seed: (int) Seed for the random number generator. Default value ``42``.

      :param inputNoise: (float) The Probability of a SDR bit to be error

    """
    def __init__(self,
                 inputFileName,
                 numBits=512,
                 numOnBits=10,
                 inputNoise=0.1,
                 SDR=None):

        Feeder.__init__(self, numBits, numOnBits, inputNoise)
        self.char_list = [char for char in open(inputFileName).read()]

        self.char_sdr = SDR
        self.readIndex = 0

    def feed(self, feedback=None, writer=None):
        inputChar = self.char_list[self.readIndex]
        sparseSDR = self.char_sdr.getNoisySDR(inputChar)
        inputSDR = self.char_sdr.getDenseFromSparse(sparseSDR)
        self.readIndex += 1
        if self.readIndex >= len(self.char_list):
            self.readIndex = 0
        return inputSDR

    def feedSparse(self, feedback=None, writer=None):
        if self.readIndex < len(self.char_list) - 1:
            self.readIndex = self.readIndex + 1
        else:
            self.readIndex = -1
        inputChar = self.char_list[self.readIndex]
        sparseSDR = set(self.char_sdr.getNoisySDR(inputChar))
        return sparseSDR

    def evaluatePrediction(self, inputChar, prediction):
        scores = [(i, self.getMatch(i, prediction)) for i in range(128)]
        # scores = [s for s in scores if s[1] > 4]
        scores.sort(key=lambda x: x[1],reverse=True)
        # print("Input: ", inputChar)
        predChars = ""
        hit = False
        for i, score in scores[:5]:
            newChar = chr(i) + " : " + str(score) + " , "
            predChars += newChar
            if inputChar == chr(i):
                hit = True
        print("Input: ", inputChar, "Prediction: ", predChars)
        if hit:
            return 1.0
        else:
            return 0.0

    def getMatch(self, i, prediction):
        return np.sum(prediction[self.char_sdr.getSDR(chr(i))].astype(np.int))
