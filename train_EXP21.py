import torch
from torch import nn
import numpy as np
import time;
localtime = time.asctime( time.localtime(time.time()) )
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

def train(net,
          train_ds,
          feeder=None,
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

    counter = 0
    mean_accuracy = 0
    mean_loss = 0

    inputs = torch.rand(numBits)
    inputs = feeder.feed(inputs)
    inputs = torch.tensor(inputs.T, dtype=torch.float32)

    for e in range(epochs):

        for i in range(len(train_dl)):
            counter += 1


            targets = feeder.feed(inputs)
            targets = torch.tensor(targets.T, dtype=torch.float32)

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
            # net.eval()
            accuracy = accuracySDR(output, targets, numOnBits)
            mean_accuracy = 0.999 * mean_accuracy + 0.001 * accuracy
            mean_loss = 0.999 * mean_loss + 0.001 * loss.item()

            inputs = targets.detach()

            # net.train()
            if counter % print_every == 0:
                print(name + ": ",
                    "Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(mean_loss),
                      "SDR Acc: {:.3f}".format(mean_accuracy))

                writer.add_scalar('perf/train_loss' , loss.item(), e)
                writer.add_scalar('perf/sdr_accuracy', train.accuracy, e)



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



