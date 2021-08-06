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
Main experiment 15
Single Layer Perceptron
"""
import numpy as np
import argparse
from train import train
from models.FC import FC
from torch.utils.data import TensorDataset
from SDR import SDR
import torch

# Some constants for the experiment
# Num of bits for the SDR input for character
# The ASCII input has 7 bits. To have sparse representation (< 2%)
# We set 512 as the number of bits and 10 as number of ON bits
NumOnBits = 10
NumBits = 512
Seed = 42
InputNoise = 0.1

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

with open(args.input, 'r') as f:
    text = f.read()

asc_chars = [chr(i) for i in range(128)]
chars = tuple(asc_chars)
int2char = dict(enumerate(chars))
char2int = {c:i for i, c in int2char.items()}

char_sdr = SDR(asc_chars,
               numBits=NumBits,
               numOnBits=NumOnBits,
               inputNoise=InputNoise)

def multi_hot_encoder(text, n_labels):
    multi_hot = np.zeros((len(text), n_labels), dtype=np.float32)
    for i, c in enumerate(text):
        sdr = char_sdr.getNoisySDR(c)
        multi_hot[i][np.array(sdr)] = 1
    return multi_hot

encoded = multi_hot_encoder(text, NumBits)

a = torch.from_numpy(encoded[0:-1, :])
b = torch.from_numpy(encoded[1:, :])

train_ds = TensorDataset(a, b)


# define and print the net
n_hidden=1024
n_layers=4

net = FC(inputDim=NumBits)
print(net)

batch_size = args.batch
seq_length = args.sequence #max length verses
n_epochs = args.epoch # start smaller if you are just testing initial behavior


# train the model
train.accuracy = 0
train(net,
      train_ds,
      epochs=n_epochs,
      batch_size=batch_size,
      lr=0.0001,
      print_every=1000,
      name=args.name,
      numBits=NumBits,
      numOnBits=NumOnBits)

