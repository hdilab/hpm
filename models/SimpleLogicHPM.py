import numpy as np
import time
import random

from models.layerHPM import NUM_PATTERN
random.seed(42)
DEBUG = False

class SimpleLogicHPM(object):
    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 lower=None,
                 printInterval=100,
                 name="layer",
                 feedbackFactor=4,
                 inputThreshold=8,
                 matchThreshold=10,
                 sizeBuffer=1000,
                 writer=None):

        super().__init__()

        self.inputMem = np.zeros((sizeBuffer, numBits))
        self.contextMem = np.zeros((sizeBuffer, numBits))
        self.inputCounter = np.zeros(sizeBuffer)
        self.contextCounter = np.zeros(sizeBuffer)
        self.predMem = {}
        self.matchThreshold = matchThreshold

        self.printInterval = printInterval
        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.feedbackFactor = feedbackFactor

        self.population = [i for i in range(numBits)]

        self.prevInputIdx = 'S'

        self.recalls = [0 for i in range(self.printInterval)]
        self.precisions = [0 for i in range(self.printInterval)]
        self.originalRecalls = [0 for i in range(self.printInterval)]
        self.originalPrecisions = [0 for i in range(self.printInterval)]
        self.accuracy = [0 for i in range(self.printInterval)]

        self.iteration = 0

        self.startTime = time.time()
        self.programStartTime = time.time()

        self.poolMask = self.buildPoolMask(self.feedbackFactor, self.numBits)

    def buildPoolMask(self, numSignal, numBits):
        eye = np.eye(numSignal)
        repNum = int(np.ceil(numBits/numSignal))
        mask = np.tile(eye, repNum)
        mask = mask[:, :numBits]
        return mask

    def feed(self, feedback={}, writer=None):
        buffer = np.zeros((self.feedbackFactor, self.numBits), dtype=bool)
        contextIdx = self.fetch(feedback, self.contextMem, self.contextCounter)
        for i in range(self.feedbackFactor):
            
            predSignal = self.predict(self.prevInputIdx, contextIdx)
            actualSignal = self.lower.feed(feedback=predSignal, writer=writer)
            actualIdx = self.fetch(actualSignal, self.inputMem, self.inputCounter)
            self.evaluate(predSignal, actualSignal, writer)
            self.update(self.prevInputIdx, contextIdx, actualIdx)
            buffer[i] = actualSignal
            self.prevInputIdx = actualIdx
            self.iteration += 1

        poolOutputSignal = self.pool(buffer)
        return poolOutputSignal

    def fetch(self, sbr, buffer, count):
        match = buffer @ sbr
        maxIdx = np.argmax(match)
        if match[maxIdx] < self.matchThreshold:
            maxIdx = np.argmin(count)
        buffer[maxIdx] = ((count[maxIdx]) * buffer[maxIdx] + sbr ) / (count[maxIdx] + 1.0)
        count[maxIdx] += 1
        return maxIdx

    def predict(self, i, c):
        if (i,c) in self.predMem:
            idx = self.samplePred(self.predMem[(i,c)])
        else:
            idx = np.argmax(self.inputCounter)
        return self.weightPred(idx)

    def update(self, i, c, a):
        if (i,c) in self.predMem:
            if a in self.predMem[(i,c)]:
                self.predMem[(i,c)][a] += 1
            else:
                self.predMem[(i,c)][a] = 1
        else:
            self.predMem[(i,c)] = {a:1} 

    def pool(self, buffer):
        output = buffer * self.poolMask
        result = output.sum(axis=0)
        return result


    def samplePred(self, logicDict):
        idxs = [k for k in logicDict.keys()]
        probs = np.array([logicDict[k] for k in idxs])
        probs = probs / probs.sum()
        pred = np.random.choice(idxs,p=probs)
        return pred


    def weightPred(self, idx):
        probs = np.ones(self.numBits) * 0.00000001
        probs += self.inputMem[idx]
        probs = probs / probs.sum()
        sparseBinary = np.random.choice(self.population,self.numOnBits,replace=False,p=probs)
        pred = self.makeBinary(sparseBinary)
        return pred

    def makeBinary(self, sparse):
        dense = np.zeros(self.numBits, dtype=bool)
        dense[sparse] = True
        return dense


    def evaluate(self, pred, target, writer):

        if type(pred) != np.ndarray:
            return

        self.precisions[self.iteration % self.printInterval], self.recalls[self.iteration % self.printInterval] = \
            self.getPrecisionRecallError(target, pred)



        if self.name == 'L1':
            charTarget = self.lower.char_sdr.getInput(target)
            charPred = self.lower.char_sdr.getInput(pred)
            originalTarget = self.lower.char_sdr.getSDR(charTarget)
            originalTarget = self.lower.char_sdr.getDenseFromSparse(originalTarget)

            self.originalPrecisions[self.iteration % self.printInterval], self.originalRecalls[
                self.iteration % self.printInterval] = \
                self.getPrecisionRecallError(originalTarget, pred)

            if charTarget == charPred:
                self.accuracy[self.iteration % self.printInterval] = 1
            else:
                self.accuracy[self.iteration % self.printInterval] = 0



        if self.iteration % self.printInterval == 0:
            meanRecall = np.mean(self.recalls)
            meanPrecision = np.mean(self.precisions)
            meanOriginalRecall = np.mean(self.originalRecalls)
            meanOriginalPrecision = np.mean(self.originalPrecisions)
            meanAccuracy = np.mean(self.accuracy)
            currentTestTime = time.time()
            trainTime = int(currentTestTime - self.startTime)
            totalTime = int((currentTestTime - self.programStartTime)/3600)
            self.startTime = currentTestTime

            print(self.name, \
                  " Iteration: ", self.iteration,
                  " R: ",  "{:.4f}".format(meanRecall),
                  " Orig-R: ",  "{:.4f}".format(meanOriginalRecall),
                  " Accuracy: ", "{:.4f}".format(meanAccuracy),
                  " Len Mem: ", len(self.predMem),
                  " Len input: ", np.sum(self.inputCounter > 0 ),
                  " Len context: ", np.sum(self.contextCounter > 0),
                  " Training Time: ", trainTime,
                  " Total Time: ", totalTime)
            writer.add_scalar('recall/origRecall'+self.name, meanOriginalRecall, self.iteration)
            # writer.add_scalar('precision/origPrecision'+self.name, meanOriginalPrecision, self.iteration)
            writer.add_scalar('recall/recall'+self.name, meanRecall, self.iteration)
            # writer.add_scalar('precision/precision'+self.name, meanPrecision, self.iteration)
            writer.add_scalar('accuracy/accuracy'+self.name, meanAccuracy, self.iteration)
            writer.add_scalar('count/predMem' + self.name, len(self.predMem), self.iteration)
            writer.add_scalar('count/inputMem' + self.name, np.sum(self.inputCounter > 0 ), self.iteration)
            writer.add_scalar('count/contextMem' + self.name, np.sum(self.contextCounter >0 ), self.iteration)

            if DEBUG:
                print('A success rate: ', self.debug['countASuccess'] / (self.debug['countA'] + 1))

    @staticmethod
    def getPrecisionRecallError(target, pred):
        newTarget = target.astype(int)
        newPred = pred.astype(int)
        intersection = (target * pred).astype(int)
        recall = intersection.sum() / (newTarget.sum()+ 0.0001)
        precision = intersection.sum() / (newPred.sum() + 0.0001)
        # if recall > 0.99:
        #     print("Hello ", self.name)
        return precision, recall


    