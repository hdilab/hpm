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
Main experiment 14
Run the LSTM
"""
import numpy as np
import argparse
from SDR import SDR
from train import train
from models.CharRNN import CharRNN

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
                    help="name of the experiment. ex) EXP-14-short" )
args = parser.parse_args()

with open(args.input, 'r') as f:
    text = f.read()

asc_chars = [chr(i) for i in range(128)]
chars = tuple(asc_chars)
int2char = dict(enumerate(chars))
char2int = {c:i for i, c in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])

# define and print the net
n_hidden=1024
n_layers=4

net = CharRNN(chars, n_hidden, n_layers, numBits=NumBits)
print(net)

batch_size = 4
seq_length = 10 #max length verses
n_epochs = 3000 # start smaller if you are just testing initial behavior

# train the model
train.accuracy = 0
train(net,
      encoded,
      epochs=n_epochs,
      batch_size=batch_size,
      seq_length=seq_length,
      lr=0.0001,
      print_every=10,
      name=args.name,
      numBits=NumBits)

