import torch
from torch import nn
import numpy as np
import time;
localtime = time.asctime( time.localtime(time.time()) )
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

def train(net,
          train_ds,
          epochs=10,
          batch_size=10,
          lr=0.001,
          clip=5,
          print_every=10,
          numBits=512,
          numOnBits=10,
          name="EXP"):

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

    writer = SummaryWriter('../runs/' + name +  localtime)

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_dl = DataLoader(train_ds, batch_size=batch_size )

    # check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')

    if (train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = numBits
    for e in range(epochs):

        for inputs, targets in train_dl:
            counter += 1



            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output = net(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output, targets)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # SDR loss
            net.eval()
            accuracy = accuracySDR(output, targets, numOnBits)
            train.accuracy = 0.999 * train.accuracy + 0.001 * accuracy



            net.train()
        if epochs % print_every == 0:


            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.4f}...".format(loss.item()),
                  "SDR Acc: {:.3f}".format(train.accuracy))

            writer.add_scalar('perf/train_loss' , loss.item(), epochs)
            writer.add_scalar('perf/sdr_accuracy', train.accuracy, epochs)



def accuracySDR(output, target, NumOnBits):
    output, target = output.cpu(), target.cpu()
    _, outputIndex = output.topk(NumOnBits, dim=1)
    _, targetIndex = target.topk(NumOnBits, dim=1)
    accuracy = np.zeros((outputIndex.shape[0]))

    for j in range(outputIndex.shape[0]):
        intersection = [i for i in outputIndex[j] if i in targetIndex[j]]
        accuracy[j] = len(intersection) * 1.0 / NumOnBits

    result = np.mean(accuracy)
    return result


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
        x = arr[:, n:n + seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


