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
Main experiment 21
MLP with BCELogit Loss for autoregressive model
NNSAE for pooler
"""
import numpy as np
import argparse
from models.HPM import HPM
from txtfeeder import TXTFeeder

# Some constants for the experiment
# Num of bits for the SDR input for character
# The ASCII input has 7 bits. To have sparse representation (< 2%)
# We set 512 as the number of bits and 10 as number of ON bits
NumOnBits = 10
NumBits = 512
Seed = 42
InputNoise = 0.0

parser = argparse.ArgumentParser()
parser.add_argument("input",
                    help="input text file you want to use")
parser.add_argument("-n", "--name",
                    help="name of the experiment. ex) EXP-14-short",
                    default="EXP")
parser.add_argument("-e", "--epoch",
                    type=int,
                    help="Number of the epoch",
                    default=1000)
parser.add_argument("-b","--batch",
                    type=int,
                    help="batch size",
                    default="64")
parser.add_argument("-s", "--sequence",
                    type=int,
                    help="sequence length",
                    default="192")
args = parser.parse_args()
n_epochs = args.epoch # start smaller if you are just testing initial behavior

L1feeder = TXTFeeder(args.input,
                     numBits=NumBits,
                     numOnBits=NumOnBits,
                     inputNoise=InputNoise)

L1 = HPM(numBits=NumBits,
         numOnBits=NumOnBits,
         lower=L1feeder,
         name="L1")

L2 = HPM(numBits=NumBits,
         numOnBits=NumOnBits,
         lower=L1,
         name="L2")

L3 = HPM(numBits=NumBits,
         numOnBits=NumOnBits,
         lower=L2,
         name="L3")


randomSDR = np.random.random(NumBits)

for i in range(n_epochs):
    L1.feed(randomSDR)
