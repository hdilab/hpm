import numpy as np

# data I/O
data = open('data/short.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_chars = {i:ch for i,ch in enumerate(chars)}

#hyperparameters
hidden_size = 100
seq_length = 25
