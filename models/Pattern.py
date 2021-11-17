import numpy as np


class Pattern(object):
    def __init__(self,
                 numOnBit=20,
                 numBit=1024,
                 threshold=10,
                 ):

        super().__init__()

        self.numOnBit = numOnBit
        self.numBit = numBit
        self.threshold = threshold

    def getIdxAndUpdate(self, signal):
        if signals not in self:
            self.binaryMatrix = signal.reshape((1,))
            self.continousMatrix = signal.reshape((1,))

            self.counts = [1]
            return 0
        else:
            match = self.binaryMatrix @ signal
            maxIdx = np.argmax(match)
            if match[maxIdx] > self.threshold:
                idx = maxIdx
                self.update(maxIdx, signal)
            else:
                newBinary = signal.reshape((1,))
                self.binaryMatrix = np.concatenate(self.binaryMatrix, newBinary, axis=0)
                self.continousMatrix = np.concatenate(self.continousMatrix, newBinary, axis=0)
                self.counts.append(1)
                idx = len(self.counts) - 1

        return idx

    def update(self, idx, signal):
        count = self.counts[idx]
        if count > 10:
            count = 10
        self.continousMatrix[idx] = ((count-1) * self.continousMatrix[idx] + signal)/count
        k = - self.numOnBit
        topk = np.argpartition(self.continousMatrix[idx], k)[k:]
        newInput = np.zeros(self.numBit, dtype=bool)
        newInput[topk] = True
        self.binaryMatrix[idx] = newInput
        self.counts[idx] += 1


