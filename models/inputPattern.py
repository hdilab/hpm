import numpy as np

from models.contextPattern import contextPattern

DEBUG = False

class inputPattern(object):
    def __init__(self,
                 position=None,
                 numBit=2048*2,
                 numOnBit=40*2,
                 threshold=8
                 ):

        super().__init__()

        self.position=position
        self.contextMatchThreshold = threshold

        self.predictionCount = 0
        self.inputHistory = np.zeros((numBit))
        self.input = np.zeros((numBit), dtype=bool)
        self.numBit = numBit
        self.numOnBit = numOnBit

        self.contextPatterns = []
        self.countUpdateInput = 0

    def replaceInput(self,input=None,context=None):
        self.input = input
        newContext = contextPattern(context=context, numBit=self.numBit, numOnBit=self.numOnBit )
        self.contextPatterns = [newContext]
        self.inputHistory = input.astype(float)
        self.countUpdateInput = 0
        self.predictionCount = 0

    def predict(self, input, context):
        isContextAvailable = False
        self.predictionCount += 1
        if len(self.contextPatterns) > 0:
            matchContext = [cP.matchContext(context) for cP in self.contextPatterns]
            maxIndex = np.argmax(matchContext)
            if matchContext[maxIndex] > self.contextMatchThreshold:
                self.activeContext = self.contextPatterns[maxIndex]
                isContextAvailable = True

        if not isContextAvailable:

            newContext = contextPattern(context=context, numOnBit=self.numOnBit)

            self.contextPatterns.append(newContext)
            self.activeContext = newContext
            if DEBUG:
                print("Context Pattern Added: Cell: ", self.position, " Contexts: ", len(self.contextPatterns))
        pred = self.activeContext.predict(context)
        return pred

    def update(self, input, context, actual):
        isMatrixChanged = self.updateInput(input)
        self.activeContext.update(context, actual)
        return isMatrixChanged

    def updateInput(self, input):
        self.countUpdateInput += 1
        count = self.countUpdateInput
        if count > 10:
            count = 10
        self.inputHistory = ((count-1)*self.inputHistory + input)/ count
        k = -self.numOnBit * 2
        topk = np.argpartition(self.inputHistory,k)[k:]
        newInput = np.zeros_like(self.input, dtype=bool)
        newInput[topk] = True
        match = self.input == newInput
        if match.all():
            return False
        else:
            self.input = newInput
            return True

