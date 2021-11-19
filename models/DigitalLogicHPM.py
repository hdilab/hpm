from models.Pattern import Pattern
import numpy as np
import time
import random
random.seed(42)
DEBUG = False

class DigitalLogicHPM(object):
    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 lower=None,
                 printInterval=100,
                 name="layer",
                 feedbackFactor=4,
                 threshold=8,
                 writer=None):

        super().__init__()
        self.inputPattern = Pattern(numBits=numBits,
                                    numOnBits=numOnBits,
                                    threshold=threshold)

        self.contextPattern = Pattern(numBits=numBits,
                                    numOnBits=numOnBits,
                                    threshold=threshold)

        self.printInterval = printInterval
        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.feedbackFactor = feedbackFactor

        self.logic = {}

        self.population = [i for i in range(numBits)]

        self.prevInputIdx = 'START'

        self.recalls = [0 for i in range(self.printInterval)]
        self.precisions = [0 for i in range(self.printInterval)]
        self.originalRecalls = [0 for i in range(self.printInterval)]
        self.originalPrecisions = [0 for i in range(self.printInterval)]
        self.accuracy = [0 for i in range(self.printInterval)]

        self.iteration = 0

        self.startTime = time.time()
        self.programStartTime = time.time()

        self.poolMask = self.buildPoolMask(self.feedbackFactor, self.numBits)
        self.debug = {'countA':0, 'countASuccess':0}

    def buildPoolMask(self, numSignal, numBits):
        eye = np.eye(numSignal)
        repNum = int(np.ceil(numBits/numSignal))
        mask = np.tile(eye, repNum)
        mask = mask[:, :numBits]
        return mask

    def feed(self, feedback={}, writer=None):
        bufferIdx = []
        for i in range(self.feedbackFactor):
            contextIdx = self.contextPattern.getIdxAndUpdate(feedback)
            predSignal = self.predict(self.prevInputIdx, contextIdx)
            actualSignal = self.lower.feed(feedback=predSignal, writer=writer)
            actualIdx = self.inputPattern.getIdxAndUpdate(actualSignal)
            self.evaluate(predSignal, actualSignal, writer)
            self.update(self.prevInputIdx, contextIdx, predSignal, actualIdx)
            bufferIdx.append(actualIdx)

            if DEBUG and self.name == 'L1':
                if self.prevInputIdx == 'START':
                    charInput = 'START'
                else:
                    charInput = self.lower.char_sdr.getInput(self.inputPattern.getSignal(self.prevInputIdx))
                charTarget = self.lower.char_sdr.getInput(actualSignal)
                charPred = self.lower.char_sdr.getInput(predSignal)
                if self.prevInputIdx == 0:
                    self.debug['countA'] += 1
                    if charPred == charTarget:
                        self.debug['countASuccess'] +=1

                if charPred != charTarget:
                    print("L1 Iter: ", self.iteration,  '(input, context) -> pred for actual :  (',
                       charInput, ' (', self.prevInputIdx, ') ,  ',  contextIdx,  ') -> ', charPred,
                      ' for ', charTarget, ' (', actualIdx, ')')

            if DEBUG and self.name != 'L1':
                if self.predIdx != actualIdx:
                    print("L2 Iter: ", self.iteration,  '(input, context) -> pred for actual :  (',
                       self.prevInputIdx, ', ', contextIdx ,') -> ', self.predIdx, ' for ', actualIdx)
            self.prevInputIdx = actualIdx
            self.iteration += 1
        poolOutputSignal = self.pool(bufferIdx)
        return poolOutputSignal

    def predict(self, input, context):
        if input not in self.logic:
            self.logic[input] = {}
        if context not in self.logic[input]:
            self.logic[input][context] = 'UNK'
            pred = 'UNK'
        elif self.logic[input][context] == 'UNK':
            pred = 'UNK'
        else:
            # pred = self.samplePred(self.logic[input][context])
            # pred = self.maxPred(self.logic[input][context])
            pred = self.weightPred(self.logic[input][context])

        return pred

    def pool(self, bufferIdx):
        bufferSignal = np.zeros((self.feedbackFactor, self.numBits), dtype=bool)
        for i, idx in enumerate(bufferIdx):
            bufferSignal[i] = self.inputPattern.getSignal(idx)
        output = bufferSignal * self.poolMask
        result = output.sum(axis=0)
        return result

    def update(self, prevInputIdx, contextIdx, predSignal, actualIdx):
        logicDict = self.logic[prevInputIdx][contextIdx]
        if type(predSignal) != np.ndarray:
            self.logic[prevInputIdx][contextIdx] = {actualIdx: 1}
        else:
            if actualIdx in logicDict:
                logicDict[actualIdx] += 1
            else:
                logicDict[actualIdx] = 1

    def samplePred(self, logicDict):
        idxs = [k for k in logicDict.keys()]
        probs = np.array([logicDict[k] for k in idxs])
        probs = probs / probs.sum()
        pred = np.random.choice(idxs,p=probs)
        return pred

    def maxPred(self, logicDict):
        max_key = max(logicDict, key=logicDict.get)
        return max_key

    def weightPred(self, logicDict):
        probs = np.ones(self.numBits) * 0.000000001
        for idx,v in logicDict.items():
            probs += self.inputPattern.getSignal(idx) * v
        probs = probs / probs.sum()
        sparseBinary = np.random.choice(self.population,self.numOnBits,replace=False,p=probs)
        pred = self.makeBinary(sparseBinary)
        return pred

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
            numInputs = len(self.inputPattern.counts)
            numContexts = len(self.contextPattern.counts)

            print(self.name, \
                  " Iteration: ", self.iteration,
                  " R: ",  "{:.4f}".format(meanRecall),
                  " Orig-R: ",  "{:.4f}".format(meanOriginalRecall),
                  " Accuracy: ", "{:.4f}".format(meanAccuracy),
                  "#Input: ", numInputs,
                  "#Context: ", numContexts,
                  " Training Time: ", trainTime,
                  " Total Time: ", totalTime)
            writer.add_scalar('recall/origRecall'+self.name, meanOriginalRecall, self.iteration)
            # writer.add_scalar('precision/origPrecision'+self.name, meanOriginalPrecision, self.iteration)
            writer.add_scalar('recall/recall'+self.name, meanRecall, self.iteration)
            # writer.add_scalar('precision/precision'+self.name, meanPrecision, self.iteration)
            writer.add_scalar('accuracy/accuracy'+self.name, meanAccuracy, self.iteration)
            writer.add_scalar('counts/input'+self.name, numInputs , self.iteration)
            writer.add_scalar('counts/context'+self.name, numContexts , self.iteration)
            writer.add_histogram('hist/input' + self.name, np.array(self.inputPattern.counts) , self.iteration)
            writer.add_histogram('hist/context' + self.name, np.array(self.contextPattern.counts) , self.iteration)
            # self.replaceCount = 0

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


    def makeBinary(self, sparse):
        dense = np.zeros(self.numBits, dtype=bool)
        dense[sparse] = True
        return dense