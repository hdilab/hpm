import numpy as np


class contextPattern(object):
    def __init__(self,
                 numOnBit=40,
                 numBit=2048,
                 context=None,
                 ):

        super().__init__()

        self.predictionCount = 0
        self.actualHistory = np.zeros((numBit))
        self.numOnBit = numOnBit
        self.numBit = numBit
        self.context = context
        self.contextHistory = self.context.astype(float)
        self.countUpdateContext = 0
        self.countUpdateActual = 0

    def replaceInput(self,
                     input=None,
                     context=None):
        self.input = input
        self.context = context
        self.predictionCount = 0

    def matchContext(self, context):
        # match = np.logical_and(self.context, context).astype(int).sum()
        match = (self.context @ context).sum()
        return match

    def predict(self, context):
        k = -self.numOnBit
        topk = np.argpartition(self.actualHistory, k)[k:]
        pred = np.zeros_like(self.actualHistory, dtype=bool)
        pred[topk] = True
        return pred

    def updateContext(self, context):
        self.countUpdateContext += 1
        count = self.countUpdateContext
        if count > 10:
            count = 10
        self.contextHistory = ((count-1)*self.contextHistory + context)/ count
        k = -self.numOnBit
        topk = np.argpartition(self.contextHistory,k)[k:]
        self.context = np.zeros_like(self.context, dtype=int)
        self.context[topk] = True

    def update(self, context, actual):
        self.updateContext(context)
        self.updateActual(actual)

    def updateActual(self, actual):
        self.countUpdateActual += 1
        count = self.countUpdateActual
        if count >10:
            count = 10
        self.actualHistory = ((count-1)*self.actualHistory + actual) / count





