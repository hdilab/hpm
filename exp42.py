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
Main experiment 42
Memoization based HPM
"""
import numpy as np
import argparse
from models.layerHPM import layerHPM
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

Seed = 42
InputNoise = 0.1
SaveInterval = 10000
versionName = "EXP-42"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="input text file you want to use for training",
                    default="data/10k.txt")
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
parser.add_argument("-p", "--print",
                    type=int,
                    help="Print interval",
                    default="100")
parser.add_argument("-x", "--bits",
                    type=int,
                    help="Multiplication factor for the NumOnBits(40) and NumBits (2048)",
                    default="1")

args = parser.parse_args()
n_epochs = args.epoch # start smaller if you are just testing initial behavior
n_layers = args.layers
NumOnBits = 40 * args.bits
NumBits = 2048 * args.bits

expName = versionName + args.name

writer = SummaryWriter('../runs2/' + expName + ' L' + str(n_layers) + ' ' + localtime, comment=expName)
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

    L1 = layerHPM(numBits=NumBits,
             numOnBits=NumOnBits,
             lower=L1feeder,
             printInterval=args.print,
             name="L1",
             feedbackFactor=2,
             contextThreshold = 16 * args.bits,
             writer=writer)

    layers = [L1]
    lowerLayer = L1

    for i in range(2, n_layers+1):
        newLayer = layerHPM(numBits=NumBits,
                        numOnBits=NumOnBits,
                        lower=lowerLayer,
                        printInterval=args.print,
                        name="L%s" % i,
                        feedbackFactor=2,
                        contextThreshold=8*args.bits,
                        writer=writer)
        layers.append(newLayer)
        lowerLayer = newLayer

    randomSDR = char_sdr.getRandomSDRDense()
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
        # torch.save(archive, archiveFilePath)

