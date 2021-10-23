MAX_NUM_DENDRITES = 128

from models.Dendrite import Dendrite

class Cell(object):
    def __init__(self,
                 index=0,
                 numBits=512,
                 numOnBits=10):
        super().__init__()

        self.dendrites = []
        self.index = index
        self.activeDendrites = []
        self.countAddDendrite = 0
        self.countPruneDendrite = 0

    def predict(self, input, context):
        self.activeDendrites = []
        for aDend in self.dendrites:
            if aDend.predict(input, context):
                self.activeDendrites.append(aDend)
                return True
        return False

    def updatePredOnActualOn(self, input, context):
        candidates = self.activeDendrites
        self.updateOldDendrites(candidates, input, context)
        for d in candidates:
            d.successCount += 1

    def updatePredOffActualOn(self, input, context):
        candidates = self.findCandidateDendrites(input, context)
        if len(candidates) > 0:
            self.updateOldDendrites(candidates, input, context)
        else:
            self.addDendrite(input, context)

    def updatePredOnActualOff(self, input, context):
        candidates = self.findActiveDendrites(input, context)
        for d in candidates:
            d.weaken(input, context)
            d.failureCount += 1
        self.pruneDendrites()

    def findActiveDendrites(self, input, context):
        activeDendrites = []
        for d in self.dendrites:
            if d.predict(input, context):
                activeDendrites.append(d)
        return activeDendrites

    def countDendrites(self):
        return len(self.dendrites)

    def getPredictionCountDendrites(self):
        total = sum([d.predictionCount for d in self.dendrites])
        mean = total / (len(self.dendrites)+0.001)
        return mean

    def getSuccessCountDendrites(self):
        total = sum([d.successCount for d in self.dendrites])
        mean = total / (len(self.dendrites)+0.001)
        return mean

    def getFailureCountDendrites(self):
        total = sum([d.failureCount for d in self.dendrites])
        mean = total / (len(self.dendrites)+0.001)
        return mean

    def updatePredOffActualOff(self, input, context):
        # for d in self.dendrites:
        #     d.decay()
        return True

    def findCandidateDendrites(self, input, context):
        candidates = {aDend for aDend in self.dendrites if aDend.isCandidate(input,context)}
        return candidates

    def updateOldDendrites(self, dendrites, input, context):
        for d in dendrites:
            d.strengthen(input, context)

    def addDendrite(self, input, context):
        if len(self.dendrites) > MAX_NUM_DENDRITES:
            self.removeWeakestDendrite()
        self.dendrites.append(Dendrite(inp=input, context=context))
        self.countAddDendrite += 1

    def removeWeakestDendrite(self):
        values = [d.sumPermanence() for d in self.dendrites]
        index_min = min(range(len(values)), key=values.__getitem__)
        self.dendrites.pop(index_min)
        self.countPruneDendrite += 1

    def pruneDendrites(self):
        for i  in reversed(range(len(self.dendrites))):
            if self.dendrites[i].hasNegativePermanence():
                self.dendrites.pop(i)
                self.countPruneDendrite += 1

    def resetCount(self):
        self.countPruneDendrite = 0
        self.countAddDendrite = 0
        for d in self.dendrites:
            d.resetCount()





