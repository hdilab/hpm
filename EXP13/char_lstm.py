# -*- coding: utf-8 -*-
"""Char-LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/hdilab/hpm/blob/master/Char-LSTM.ipynb
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
Colab = False
NumOnBits = 10
NumBits = 512
Seed = 42

if Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    with open('/content/drive/My Drive/Colab/data/1342.txt','r') as f:
        text = f.read()
else:
    with open('data/1342.txt','r') as f:
        text = f.read()

text[:100]

asc_chars = [chr(i) for i in range(128)]
chars = tuple(asc_chars)
int2char = dict(enumerate(chars))
char2int = {c:i for i, c in int2char.items()}

encoded = np.array([char2int[ch] for ch in text])
encoded[:100]

"""
SDR class
Handles issues with SDR
Given a char input, generate SDR
"""

import random


class SDR(object):
    """
      Class implementing the SDR.

      :param input_list: (List) List for input_values.
            For ASCII it will be [chr(0), chr(1), ... chr(127)]

      :param numBits: (int) Number of bits for SDR. Default value ``512``

      :param numOnBits: (int) Number of Active bits for SDR. Default value ``10``.
            It is 2% sparcity for 512 bit

      :param seed: (int) Seed for the random number generator. Default value ``42``.
    """

    def __init__(self,
                 input_list,
                 numBits=512,
                 numOnBits=10,
                 seed=42,
                 inputNoise=0.1):

        random.seed(seed)
        self.population = [i for i in range(numBits)]
        self.numOnBits = numOnBits
        self.inputNoise = inputNoise
        self.sdr_dict = {i:random.sample(self.population, numOnBits) for i in input_list}


    def getSDR(self, input):
        return self.sdr_dict[input]


    def getNoisySDR(self, input):
        inputSDR = self.sdr_dict[input]
        inputSDR = [i for i in inputSDR if random.random() > self.inputNoise]
        noise = random.sample(self.population, int(self.numOnBits * self.inputNoise))
        return inputSDR + noise



    def getInput(self, sdr):
        """
        Need to implement the function which returns the corresponding input from SDR
        This requires a probabilistic approach. Count the number of overlapping bit and nonoverlapping field.
        """
        return 0

    def getCollisionProb(self, n, a, s, theta):
        """
        Calculating the probability for the cases where more than theta synapses are activated
        for different cell activation pattern
        :param n: Number of cells
        :param a: Number of active cells
        :param s: Number of synapses
        :param theta: Threshold for the dendritic activation
        :return: The probability where dendritic activation for the different cell activation pattern
        """
        numerator = 0
        for b in range(theta, s+1):
            numerator += combinatorial(s, b) * combinatorial(n-s, a-b)

        denominator = combinatorial(n, a)

        return numerator*1.0/denominator

    def getRandomSDR(self):
        noise = random.sample(self.population, numOnBits)
        return noise


def combinatorial(a,b):
    return factorial(a)*1.0/factorial(a-b)/factorial(a)

def factorial(a):
    if a == 1:
        return 1
    else:
        return a*factorial(a-1)

char_sdr = SDR(asc_chars,
                numBits=NumBits,
                numOnBits=NumOnBits,
                seed=Seed,
                inputNoise=0.1)

def one_hot_encoder(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1. 
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

def multi_hot_encoder(arr, n_labels):
    multi_hot = np.zeros((arr.shape[0], arr.shape[1], n_labels), dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sdr = char_sdr.getNoisySDR(int2char[arr[i][j]])
            multi_hot[i][j][np.array(sdr)] = 1  
    return multi_hot

test_seq = np.array([[3,5,1]])
one_hot=one_hot_encoder(test_seq, 8)
multi_hot = multi_hot_encoder(test_seq, 512)

print (test_seq.shape)

print(one_hot)

print(multi_hot.shape)

sdr = char_sdr.getNoisySDR(int2char[1])
a = np.zeros((3,512))
a[1][np.array(sdr)] = 1
print (sdr)
print(np.argwhere(multi_hot[0,2]>0))

def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    n_batches = len(arr) // batch_size_total
    
    arr = arr[:n_batches * batch_size_total]
    arr = arr.reshape((batch_size, -1))
    
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x) 
        try:
            y[:, :-1], y[:, -1] = x[:,1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:,1:], arr[:,0] 
        yield x, y

batches = get_batches(encoded, 8, 50)
x, y = next(batches)

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')

class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=612, n_layers=4, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch:ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(NumBits, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, NumBits)
        
    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x,hidden)
        
        out = self.dropout(r_output)
        
        out = out.contiguous().view(-1, self.n_hidden)
        
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_chars = NumBits
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            x = multi_hot_encoder(x, n_chars)
            y = multi_hot_encoder(y, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length, NumBits))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = multi_hot_encoder(x, n_chars)
                    y = multi_hot_encoder(y, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length, NumBits))
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

# define and print the net
n_hidden=1024
n_layers=4

net = CharRNN(chars, n_hidden, n_layers)
print(net)

batch_size = 64
seq_length = 160 #max length verses
n_epochs = 50 # start smaller if you are just testing initial behavior

# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

model_dante = 'rnn_20_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_dante, 'wb') as f:
    torch.save(checkpoint, f)

def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encoder(x, len(net.chars))
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        # apply softmax to get p probabilities for the likely next character giving x
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        # get top characters
        # considering the k most probable characters with topk method
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h

def sample(net, size, prime='Il', top_k=None):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(net, 1000, prime='This ', top_k=5))

y

x, y = next(batches)
print(x,y)

