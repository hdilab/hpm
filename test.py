import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

bias = torch.randn(512)
input = torch.randn(1,512)
weight = torch.randn(512,512)

test = torch.addmm(bias, input, weight.t())
print(test)

biasN = bias.detach().numpy()
inputN = input.detach().numpy()
weightN = weight.detach().numpy()

testN = np.matmul(inputN, weightN.T) + biasN
# print(testN)

error = test.detach().numpy() - testN

print(np.sum(error))

import pickle

with open('data.exp15', 'wb') as f:
    pickle.dump({'biasN': biasN,
                 'inputN': inputN,
                 'weightN': weightN,
                 'testN': testN}, f)

