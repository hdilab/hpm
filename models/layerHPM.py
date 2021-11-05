from models.inputPattern import inputPattern
import numpy as np
import time
import random
random.seed(42)
NUM_PATTERN = 4096
MATCH_THRESHOLD = 40
DEBUG = False
DEBUG_ERROR_ONLY = False

class layerHPM(object):
    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 lower=None,
                 printInterval=100,
                 name="layer",
                 feedbackFactor=4,
                 writer=None):

        super().__init__()
        self.patterns = []

        self.printInterval = printInterval
        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.feedbackFactor = feedbackFactor

        self.patternMatrix = np.zeros((NUM_PATTERN,numBits*2))
        self.patterns = [inputPattern(position=i) for i in range(NUM_PATTERN)]

        population = range(numBits)

        self.pred = self.makeBinary(random.sample(population, numOnBits))
        context = self.makeBinary(random.sample(population, numOnBits))
        self.prevPrevActual = self.lower.feed(feedback=self.pred, writer=writer)
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

        self.poolMask = self.buildPoolMask(self.feedbackFactor, self.numBits)

    def buildPoolMask(self, numSignal, numBits):
        eye = np.eye(numSignal)
        repNum = int(np.ceil(numBits/numSignal))
        mask = np.tile(eye, repNum)
        mask = mask[:, :numBits]
        return mask

    def feed(self, feedback={}, writer=None):
        buffer = np.zeros((self.feedbackFactor,self.numBits), dtype=bool)
        if self.name == 'L1' and (DEBUG or DEBUG_ERROR_ONLY):
            charContext = self.analyzeContext(feedback)
        for i in range(self.feedbackFactor):
            context = feedback
            input = np.hstack((self.prevPrevActual, self.prevActual))
            self.pred = self.predict(input, context)
            self.actual = self.lower.feed(feedback=self.pred, writer=writer)

            if ( DEBUG or DEBUG_ERROR_ONLY) and self.name == 'L1':
                charPrevInput = self.lower.char_sdr.getInput(self.prevPrevActual)
                charInput = self.lower.char_sdr.getInput(self.prevActual)
                charTarget = self.lower.char_sdr.getInput(self.actual)
                charPred = self.lower.char_sdr.getInput(self.pred)
                numContext = np.array([len(p.contextPatterns) for p in self.patterns])
                sumNumContext = np.sum(numContext)
                if charPred != charTarget and DEBUG_ERROR_ONLY:
                    print("L1 Iter: ", self.iteration, 'Pattern: ', self.replaceCount, ' Context: ', sumNumContext,  '(input[prev, curr], context) -> pred for actual :  ([',
                      charPrevInput, charInput, ', ', charContext[i], ') -> ', charPred, ' for ', charTarget)

            if (DEBUG) and self.name == 'L2':
                charPrevInput = self.lower.analyzeContext(self.prevPrevActual)
                charInput = self.lower.analyzeContext(self.prevActual)
                charTarget = self.lower.analyzeContext(self.actual)
                charPred = self.lower.analyzeContext(self.pred)
                # charContext = self.analyzeContext(context)
                numContext = np.array([len(p.contextPatterns) for p in self.patterns])
                sumNumContext = np.sum(numContext)
                if charPred != charTarget and DEBUG_ERROR_ONLY:
                    print("L2 Iter: ", self.iteration, 'Pattern: ', self.replaceCount, ' Context: ', sumNumContext,  '(input[prev, curr], context) -> pred for actual :  ([',
                      charPrevInput, charInput, ', ', context.nonzero()[0][:4], ') -> ', charPred, ' for ', charTarget)

            self.evaluate(self.pred, self.actual, writer)
            self.update(input, context, self.actual, writer=writer)
            buffer[i] = self.actual
            self.prevPrevActual = self.prevActual
            self.prevActual = self.actual
            self.iteration += 1
        poolOutput = self.pool(buffer, writer)
        # context = poolOutput
        return poolOutput

    def analyzeContext(self, context):
        contextReshape = context.reshape((-1,1))
        zeroPadding = np.zeros((self.numBits, self.feedbackFactor-1))
        reconstruct = np.hstack((contextReshape, zeroPadding))
        recon = reconstruct.flatten()
        recon = np.split(recon,2)
        charContext = [self.lower.char_sdr.getInput(r) for r in recon]
        return charContext

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
            if DEBUG and self.name == 'L1' :
                splitInput = np.split(input, 2)
                prevPrevInput = splitInput[0]
                prevInput = splitInput[1]
                print("Replace Worst Pattern for ", self.lower.char_sdr.getInput(prevPrevInput), self.lower.char_sdr.getInput(prevInput), " Pos: ", worstPattern.position)
        newContext = context.astype(int)
        pred = self.activePattern.predict(input, newContext)
        return pred

    def findWorstPattern(self):
        performances = [p.predictionCount for p in self.patterns]
        worst = np.argmin(performances)
        return self.patterns[worst]

    def pool(self, buffer, writer):
        output = buffer * self.poolMask
        result = output.sum(axis=0)
        return result

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