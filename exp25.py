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
Main experiment 24
Build Autoencoder model
Use pytorch to build a standard autoencoder
"""
import numpy as np
import argparse
from models.HPM import HPM
from txtfeeder import TXTFeeder
from models.AE import autoencoder
import torch
import time
import os
from torch import nn
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
TestInterval = 1000
learning_rate = 1e-3
inputNoise = 0.1
numTestCharacter = 10000
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="input text file you want to use for training",
                    default="data/5M-train.txt")
parser.add_argument("-t", "--test",
                    help="input text file you want to use for testing",
                    default="data/500K-test.txt")
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
parser.add_argument("-l", "--layers",
                    type=int,
                    help="Number of layers",
                    default="3")
parser.add_argument("-f", "--file",
                    help="path to the saved model",
                    default="NA")

args = parser.parse_args()
n_epochs = args.epoch # start smaller if you are just testing initial behavior
n_layers = args.layers



writer = SummaryWriter('../runs/exp-25-' + ' ' + localtime, comment='EXP-25 Layer'+str(n_layers))
os.makedirs('./save', exist_ok=True)
archiveFilePath = './save/exp-25-' + str(n_layers) + ' ' + localtime + '.pt'



asc_chars = [chr(i) for i in range(128)]
char_sdr = SDR(asc_chars,
                numBits=NumBits,
                numOnBits=NumOnBits,
                inputNoise=inputNoise,
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
criterion = nn.BCEWithLogitsLoss()
if args.file == "NA":
    # Start new training

    AE = autoencoder(numBits=NumBits)

    optimizer = torch.optim.Adam(AE.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)

else:
    AE = autoencoder(numBits=NumBits)
    archive = torch.load(args.file)
    AE.load_state_dict(archive['AE'])
    previousEpoch = archive['epoch']
    optimizer = torch.optim.Adam(AE.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    optimizer.load_state_dict(archive['optimizer'])
    loss = archive['loss']

trainLoss = 0.0
for i in range(n_epochs):
    input = []
    for _ in range(4):
        signal = trainFeeder.feed()
        signal = np.squeeze(signal).tolist()
        input.extend(signal)
    input = torch.tensor(input)
    input = torch.reshape(input, (1,-1))
    recon = AE(input)
    loss = criterion(recon, input)
    trainLoss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % SaveInterval == 0:
        archive = {'AE': AE.state_dict(),
                   'epoch': i,
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': loss}
        torch.save(archive, archiveFilePath)
        print('Save file: epoch [{}/{}], loss:{:.4f}'
              .format(i + 1, n_epochs, loss.item()))

    if i % TestInterval == 0:
        testLoss = 0.0;
        with torch.no_grad():
            for j in range(numTestCharacter):
                input = []
                for _ in range(4):
                    signal = trainFeeder.feed()
                    signal = np.squeeze(signal).tolist()
                    input.extend(signal)
                input = torch.tensor(input)
                input = torch.reshape(input, (1, -1))
                recon = AE(input)
                loss = criterion(recon, input)
                testLoss += loss

        testLoss /= numTestCharacter
        trainLoss /= TestInterval
        print('epoch [{}/{}], Test Loss:{:.6f},  Train Loss:{:.6f}'
                  .format(i + 1, n_epochs, testLoss, trainLoss ))
        writer.add_scalar('test loss/SimpleAE-BCE', testLoss, i)
        writer.add_scalar('train loss/SimpleAE-BCE', trainLoss, i)
        trainLoss = 0.0
        writer.add_histogram('AE.decoder.linear2.weight',AE.decoder[2].weight, i)
        writer.add_histogram('AE.decoder.linear2.bias', AE.decoder[2].bias, i)
        writer.add_histogram('AE.output', recon, i)
        writer.add_histogram('AE.input', input, i)
        topValues, topIndices = torch.topk(recon, NumOnBits*8)
        writer.add_histogram('AE.output.TopValues', topValues,i)

