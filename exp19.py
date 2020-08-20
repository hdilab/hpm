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
Main experiment 19
Use NNSAE for autoregressing char SDR
Single layer
"""

from models.NNSAR import NNSAR
from txtfeeder import TXTFeeder
import time
import numpy as np
import random
from SDR import SDR

start = time.time()

# Some constants for the experiment
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# Num of bits for the SDR input for character
# The ASCII input has 7 bits. To have sparse representation (< 2%)
# We set 512 as the number of bits and 10 as number of ON bits
NUM_SDR_BIT = 512
NUM_SDR_ON_BIT = 10
INPUT_NOISE = 0.0

L1feeder = TXTFeeder("data/nonoverlapping.txt",
                     numBits=NUM_SDR_BIT,
                     numOnBits=NUM_SDR_ON_BIT,
                     inputNoise=INPUT_NOISE)

L1 = NNSAR(inputDim=NUM_SDR_BIT,
           hiddenDim=NUM_SDR_BIT//4,
           lower=L1feeder,
           name="L1"
           )

context = np.zeros(NUM_SDR_BIT)
for i in range(1000000):
    L1.feed(context)