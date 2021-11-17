from models.Pattern import Pattern
import numpy as np
import time
import random
random.seed(42)
DEBUG = False
DEBUG_ERROR_ONLY = False

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
                                    threhold=threshold)

        self.contextPattern = Pattern(numBits=numBits,
                                    numOnBits=numOnBits,
                                    threhold=threshold)

        self.printInterval = printInterval
        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.feedbackFactor = feedbackFactor

        self.logic = {}

        population = range(numBits)

        self.prevInputIdx = 0

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
        bufferIdx = []
        for i in range(self.feedbackFactor):
            contextIdx = self.contextPattern.getIdxAndUpdate(feedback)
            self.predIdx = self.predict(self.prevInputIdx, contextIdx)
            predSignal = self.inputPattern.getSignal(self.predIdx)
            actualSignal = self.lower.feed(feedback=predSignal, writer=writer)
            actualIdx = self.inputPattern.getIdxAndUpdate(actualSignal)
            self.evaluate(predSignal, actualSignal, writer)
            self.update(self.prevInputIdx, contextIdx, self.predIdx, actualIdx)
            bufferIdx.append(actualIdx)
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
        else:
            pred = self.samplePred(self.logic[input][context])
        return pred

    def pool(self, bufferIdx, writer):
        bufferSignal = np.zeros((self.feedbackFactor, self.numBits), dtype=bool)
        for i, idx in enumerate(bufferIdx):
            bufferSignal[i] = self.inputPattern.getSignal(idx)
        output = bufferSignal * self.poolMask
        result = output.sum(axis=0)
        return result

    def update(self, prevInputIdx, contextIdx, predIdx, actualIdx):
        logicDict = self.logic[prevInputIdx][contextIdx]
        if predIdx == 'UNK':
            logicDict = {actualIdx: 1}
        else:
            if actualIdx in logicDict:
                logicDict[actualIdx] += 1
            else:
                logicDict[actualIdx] = 1

    def samplePred(self, logicDict):
        idxs = [k for k in logicDict.keys()]
        probs = np.array([logicDict[k] for k in idxs])
        probs /= probs.sum()
        pred = np.random.choice(idxs,p=probs)
        return pred

    def evaluate(self, pred, target, writer):

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
            # else:
            #     print("Error: Target ", charTarget, " Pred ",  charPred)


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
            numContext = np.array([len(p.contextPatterns) for p in self.patterns])
            meanNumContext = np.sum(numContext)



            # for c in self.cells:
            #     c.resetCount()

            print(self.name, \
                  " Iteration: ", self.iteration,
                  " R: ",  "{:.4f}".format(meanRecall),
                  " P: ",  "{:.4f}".format(meanPrecision),
                  " Orig-R: ",  "{:.4f}".format(meanOriginalRecall),
                  " Orig-P: ",  "{:.4f}".format(meanOriginalPrecision),
                  " Accuracy: ", "{:.4f}".format(meanAccuracy),
                  " Context: ", "{:.1f}".format(meanNumContext),
                  " Replace: ", "{}".format(self.replaceCount),
                  " Training Time: ", trainTime,
                  " Total Time: ", totalTime)
            writer.add_scalar('recall/origRecall'+self.name, meanOriginalRecall, self.iteration)
            writer.add_scalar('precision/origPrecision'+self.name, meanOriginalPrecision, self.iteration)
            writer.add_scalar('recall/recall'+self.name, meanRecall, self.iteration)
            writer.add_scalar('precision/precision'+self.name, meanPrecision, self.iteration)
            writer.add_scalar('accuracy/accuracy'+self.name, meanAccuracy, self.iteration)
            writer.add_scalar('counts/context'+self.name, meanNumContext, self.iteration)
            writer.add_scalar('counts/replace'+self.name, self.replaceCount , self.iteration)
            writer.add_histogram('hist/context' + self.name, numContext , self.iteration)
            # self.replaceCount = 0


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