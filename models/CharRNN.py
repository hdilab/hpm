from torch import nn
import torch

class CharRNN(nn.Module):
    def __init__(self,
                 tokens,
                 n_hidden=612,
                 n_layers=4,
                 drop_prob=0.5,
                 lr=0.001,
                 numBits=512):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.lstm = nn.LSTM(numBits, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, numBits)

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)

        out = self.dropout(r_output)

        out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden