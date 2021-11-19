import numpy as np
import random

class Pattern(object):
    def __init__(self,
                 numOnBits=20,
                 numBits=1024,
                 threshold=10,
                 ):

        super().__init__()

        self.numOnBits = numOnBits
        self.numBits = numBits
        self.threshold = threshold
        self.counts = []
        self.binaryMatrix = []
        self.continousMatrix = []
        population = range(numBits)
        self.UNK = self.makeBinary(random.sample(population, numOnBits))


    def makeBinary(self, sparse):
        dense = np.zeros(self.numBits, dtype=bool)
        dense[sparse] = True
        return dense

    def getIdxAndUpdate(self, signal):
        if type(signal) != np.ndarray:
            return 'UNK'
        if len(self.binaryMatrix) == 0 :
            self.binaryMatrix = signal.reshape((1,-1)).astype(int)
            self.continousMatrix = signal.reshape((1,-1)).astype(float)
            self.counts.append(1)
            return 0
        else:
            match = self.binaryMatrix @ signal
            maxIdx = np.argmax(match)
            if match[maxIdx] > self.threshold:
                idx = maxIdx
                self.update(maxIdx, signal)
            else:
                newBinary = signal.reshape((1,-1)).astype(int)
                self.binaryMatrix = np.concatenate((self.binaryMatrix, newBinary), axis=0)
                self.continousMatrix = np.concatenate((self.continousMatrix, newBinary), axis=0)
                self.counts.append(1)
                idx = len(self.counts) - 1

        return idx

    def getSignal(self, idx):
        if idx == 'UNK':
            return self.UNK
        else:
            return self.binaryMatrix[idx]

    def update(self, idx, signal):
        count = self.counts[idx] + 1
        if count > 10:
            count = 10
        self.continousMatrix[idx] = ((count-1) * self.continousMatrix[idx] + signal)/count
        k = - self.numOnBits
        topk = np.argpartition(self.continousMatrix[idx], k)[k:]
        newInput = np.zeros(self.numBits, dtype=int)
        newInput[topk] = True
        self.binaryMatrix[idx] = newInput
        self.counts[idx] += 1


