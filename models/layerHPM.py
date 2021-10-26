from models.inputPattern import inputPattern
import numpy as np
import time
import random
random.seed(42)
NUM_PATTERN = 1024
MATCH_THRESHOLD = 20


class layerHPM(object):
    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 lower=None,
                 printInterval=100,
                 name="layer",
                 writer=None):

        super().__init__()
        self.patterns = []

        self.printInterval = printInterval
        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits

        self.patternMatrix = np.zeros((NUM_PATTERN,numBits))
        self.patterns = [inputPattern(position=i) for i in range(NUM_PATTERN)]

        population = range(numBits)

        self.pred = self.makeBinary(random.sample(population, numOnBits))
        self.context = self.makeBinary(random.sample(population, numOnBits))
        self.prevActual = self.lower.feed(feedback=self.pred, writer=writer)

        self.recalls = [0 for i in range(self.printInterval)]
        self.precisions = [0 for i in range(self.printInterval)]
        self.originalRecalls = [0 for i in range(self.printInterval)]
        self.originalPrecisions = [0 for i in range(self.printInterval)]
        self.accuracy = [0 for i in range(self.printInterval)]
        self.replaceCount = 0

        self.iteration = 0

        self.startTime = time.time()
        self.programStartTime = time.time()

    def feed(self, feedback={}, writer=None):
        buffer = np.zeros((4,self.numBits), dtype=bool)
        for i in range(4):
            self.actual = self.lower.feed(feedback=self.pred, writer=writer)
            # self.context = self.context | feedback
            self.pred = self.predict(self.prevActual, self.context)
            # charInput = self.lower.char_sdr.getInput(self.prevActual)
            # charTarget = self.lower.char_sdr.getInput(self.actual)
            # charPred = self.lower.char_sdr.getInput(self.pred)
            # contextSum = self.context.nonzero()[0][:4]
            # print("Iter: ", self.iteration, ' (input, context) -> pred for actual :  (',
            #       charInput, ', ', contextSum, ') -> ', charPred, ' for ', charTarget)
            self.evaluate(self.pred, self.actual, writer)
            self.update(self.prevActual, self.context, self.actual, writer=writer)
            buffer[i] = self.prevActual
            self.prevActual = self.actual
            self.iteration += 1
        poolOutput = self.pool(buffer, writer)
        self.context = poolOutput
        return poolOutput

    def predict(self, input, context):
        match = self.patternMatrix @ input
        maxIndex = np.argmax(match)
        if match[maxIndex] > MATCH_THRESHOLD:
            self.activePattern = self.patterns[maxIndex]
        else:
            worstPattern = self.findWorstPattern()
            worstPattern.replaceInput(input, context)
            self.patternMatrix[worstPattern.position] = input
            self.activePattern = worstPattern
            self.replaceCount += 1
            print("Replace Worst Pattern for ", self.lower.char_sdr.getInput(input), " Pos: ", worstPattern.position)
        newContext = context.astype(int)
        pred = self.activePattern.predict(input, newContext)
        return pred

    def findWorstPattern(self):
        performances = [p.predictionCount for p in self.patterns]
        worst = np.argmin(performances)
        return self.patterns[worst]

    def pool(self, buffer, writer):
        output = buffer.reshape((self.numBits, 4))
        output = output[:,0]
        # output = output.astype(bool)
        return output

    def update(self, input, context, actual, writer=None):
        isMatrixChanged = self.activePattern.update(input, context, actual)
        if isMatrixChanged:
            self.patternMatrix[self.activePattern.position] = self.activePattern.input

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
            self.replaceCount = 0


    @staticmethod
    def getPrecisionRecallError(target, pred):
        newTarget = target.astype(int)
        newPred = pred.astype(int)
        intersection = (target & pred).astype(int)
        recall = intersection.sum() / (newTarget.sum()+ 0.0001)
        precision = intersection.sum() / (newPred.sum() + 0.0001)
        # if recall > 0.99:
        #     print("Hello ", self.name)
        return precision, recall


    def makeBinary(self, sparse):
        dense = np.zeros(self.numBits, dtype=bool)
        dense[sparse] = True
        return dense