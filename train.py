import torch
from torch import nn
import numpy as np
import time;
localtime = time.asctime( time.localtime(time.time()) )
from SDR import SDR
from tensorboardX import SummaryWriter

NumOnBits = 10
NumBits = 512
Seed = 42
InputNoise = 0.1






def train(net, data,
          epochs=10,
          batch_size=10,
          seq_length=50,
          lr=0.001,
          clip=5,
          val_frac=0.1,
          print_every=10,
          numBits=512,
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

    asc_chars = [chr(i) for i in range(128)]
    chars = tuple(asc_chars)
    int2char = dict(enumerate(chars))
    char2int = {c: i for i, c in int2char.items()}
    char_sdr = SDR(asc_chars,
                   numBits=NumBits,
                   numOnBits=NumOnBits,
                   seed=Seed,
                   inputNoise=InputNoise)

    def multi_hot_encoder(arr, n_labels):
        multi_hot = np.zeros((arr.shape[0], arr.shape[1], n_labels), dtype=np.float32)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sdr = char_sdr.getNoisySDR(int2char[arr[i][j]])
                multi_hot[i][j][np.array(sdr)] = 1
        return multi_hot

    writer = SummaryWriter('../runs/' + name +  localtime)

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

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
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = multi_hot_encoder(x, n_chars)
            y = multi_hot_encoder(y, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length, NumBits))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # SDR loss
            accuracy = accuracySDR(output, targets.view(batch_size * seq_length, NumBits))
            train.accuracy = 0.99 * train.accuracy + 0.01 * accuracy

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                accuacy = accuracySDR(output, targets.view(batch_size * seq_length, NumBits))
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
                    if (train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length, NumBits))

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterationg through validation data

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)),
                      "SDR Acc: {:.3f}".format(train.accuracy))

                writer.add_scalar('perf/train_loss' , loss.item(), counter)
                writer.add_scalar('perf/val_loss', np.mean(val_losses), counter)
                writer.add_scalar('perf/sdr_accuracy', train.accuracy, counter)

    def multi_hot_encoder(arr, n_labels):
        multi_hot = np.zeros((arr.shape[0], arr.shape[1], n_labels), dtype=np.float32)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sdr = char_sdr.getNoisySDR(int2char[arr[i][j]])
                multi_hot[i][j][np.array(sdr)] = 1
        return multi_hot


def accuracySDR(output, target):
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


