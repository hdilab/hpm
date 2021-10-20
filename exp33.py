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
Main experiment 32
HTM with kWTA AE
"""
import numpy as np
import argparse
from models.HPM import HPM
from txtfeeder import TXTFeeder
import torch
import time
import os
from SDR import SDR

localtime = time.asctime(time.localtime(time.time()))
from torch.utils.tensorboard import SummaryWriter

# Some constants for the experiment
# Num of bits for the SDR input for character
# The ASCII input has 7 bits. To have sparse representation (< 2%)
# We set 512 as the number of bits and 10 as number of ON bits
NumOnBits = 10
NumBits = 512
Seed = 42
InputNoise = 0.1
SaveInterval = 10000
TestInterval = 50000
versionName = "EXP-33"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="input text file you want to use for training",
                    default="data/5M-train.txt")
parser.add_argument("-t", "--test",
                    help="input text file you want to use for testing",
                    default="data/500K-test.txt")
parser.add_argument("-n", "--name",
                    help="name of the experiment. ex) EXP-14-short",
                    default="R1")
parser.add_argument("-e", "--epoch",
                    type=int,
                    help="Number of the epoch",
                    default=10000000)
parser.add_argument("-b","--batch",
                    type=int,
                    help="batch size",
                    default="64")
parser.add_argument("-s", "--sequence",
                    type=int,
                    help="sequence length",
                    default="192")
parser.add_argument("-l", "--layers",
                    type=int,
                    help="Number of layers",
                    default="1")
parser.add_argument("-f", "--file",
                    help="path to the saved model",
                    default="NA")

args = parser.parse_args()
n_epochs = args.epoch # start smaller if you are just testing initial behavior
n_layers = args.layers

expName = versionName + args.name

writer = SummaryWriter('../runs/' + expName + ' L' + str(n_layers) + ' ' + localtime, comment=expName)
os.makedirs('./save', exist_ok=True)
archiveFilePath = './save/' + expName + ' L' + str(n_layers) + ' ' + localtime + '.pt'

asc_chars = [chr(i) for i in range(128)]
char_sdr = SDR(asc_chars,
                numBits=NumBits,
                numOnBits=NumOnBits,
                inputNoise=InputNoise,
                seed=42);
trainFeeder = TXTFeeder(args.input,
                     numBits=NumBits,
                     numOnBits=NumOnBits,
                     inputNoise=InputNoise,
                     SDR=char_sdr)
testFeeder = TXTFeeder(args.test,
                     numBits=NumBits,
                     numOnBits=NumOnBits,
                     inputNoise=InputNoise,
                     SDR=char_sdr)

if args.file == "NA":
    # Start new training
    L1feeder = TXTFeeder(args.input,
                         numBits=NumBits,
                         numOnBits=NumOnBits,
                         inputNoise=InputNoise,
                         SDR=char_sdr)

    L1 = HPM(numBits=NumBits,
             numOnBits=NumOnBits,
             lower=L1feeder,
             name="L1")

    layers = [L1]
    lowerLayer = L1

    for i in range(2, n_layers+1):
        newLayer = HPM(numBits=NumBits,
                        numOnBits=NumOnBits,
                        lower=lowerLayer,
                        name="L%s" % i)
        layers.append(newLayer)
        lowerLayer = newLayer

    randomSDR = torch.rand((1, NumBits))
else:
    archive = torch.load(args.file)
    layers = archive['layers']
    randomSDR = archive['randomSDR']
    L1feeder = archive['L1feeder']
    n_layers = len(layers)


for i in range(n_epochs):
    layers[n_layers-1].feed(randomSDR, writer)
    if i % SaveInterval == 0:
        archive = {'layers': layers,
                   'randomSDR': randomSDR,
                   'L1feeder': L1feeder}
        torch.save(archive, archiveFilePath)

