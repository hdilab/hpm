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
Main experiment 1
Read the Pride and Prejudice from Jane Austin and use if to train HPM
Calculate the LogLikelyhood error rate
"""

from models.GBHFM import HeterarchicalPredictionMemory
from txtfeeder import TXTFeeder
import time
import pickle
import random

LOAD_MODEL = False
SAVE_MODEL = False

start = time.time()

# Some constants for the experiment
RANDOM_SEED = 42
# Num of bits for the SDR input for character
# The ASCII input has 7 bits. To have sparse representation (< 2%)
# We set 512 as the number of bits and 10 as number of ON bits
NUM_SDR_BIT = 512
NUM_SDR_ON_BIT = 10
INPUT_NOISE = 0.1

L1feeder = TXTFeeder("data/nonoverlapping.txt",
                     numBits=NUM_SDR_BIT,
                     numOnBits=NUM_SDR_ON_BIT,
                     seed=RANDOM_SEED,
                     inputNoise=INPUT_NOISE)

if LOAD_MODEL:
    f = open("model.pkl",'rb')
    L1 = pickle.load(f)
    f.close()
else:
    L1 = HeterarchicalPredictionMemory(sizeSDR=NUM_SDR_BIT,
                                       numOnBits=NUM_SDR_ON_BIT,
                                       seed=RANDOM_SEED,
                                       lower=L1feeder,
                                       name="L1"
                                       )
    L2 = HeterarchicalPredictionMemory(sizeSDR=NUM_SDR_BIT,
                                       numOnBits=NUM_SDR_ON_BIT,
                                       seed=RANDOM_SEED,
                                       lower=L1,
                                       name="L2"
                                       )

    L3 = HeterarchicalPredictionMemory(sizeSDR=NUM_SDR_BIT,
                                       numOnBits=NUM_SDR_ON_BIT,
                                       seed=RANDOM_SEED,
                                       lower=L2,
                                       name="L3"
                                       )

population = [i for  i in range(NUM_SDR_BIT)]
randomSDR = random.sample(population, NUM_SDR_ON_BIT)

for i in range(10000):
    if i%1000 == 999 and SAVE_MODEL:
        pickle.dump(L1, open("model.pkl","wb"))
        print("###### iteration ######", i)
        end = time.time()
        elasped = end - start
        print(time.strftime("%H:%M:%S", time.gmtime(elasped)))

    L3.feed(randomSDR)


