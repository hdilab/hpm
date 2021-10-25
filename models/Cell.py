MAX_NUM_DENDRITES = 128
MAX_NUM_BABYDENDRITES = 128

from models.Dendrite import Dendrite
import random

class Cell(object):
    def __init__(self,
                 index=0,
                 numBits=512,
                 numOnBits=10):
        super().__init__()

        self.dendrites = []
        self.index = index
        self.activeDendrites = []
        self.candidates = []
        self.babyDendrites = []
        self.countAddDendrite = 0
        self.countPruneDendrite = 0
        self.numOnBits = numOnBits
        self.countAddBabyDendrite = 0
        self.countPruneBabyDendrite = 0

    def predict(self, input, context):
        self.activeDendrites = []
        self.candidates = []
        for aDend in self.dendrites:
            if aDend.isCandidate(input, context):
                if aDend.predict(input, context):
                    self.activeDendrites.append(aDend)
                else:
                    self.candidates.append(aDend)

        if len(self.activeDendrites) > 0:
            return True
        else:
            return False

    def updatePredOnActualOn(self, input, context):
        candidates = self.activeDendrites
        self.updateOldDendrites(candidates, input, context)
        for d in candidates:
            d.successCount += 1

    def updatePredOffActualOn(self, input, context):
        candidates = self.candidates
        if len(candidates) > 0:
            self.updateOldDendrites(candidates, input, context)
        else:
            self.checkBabies(input, context)

    def updatePredOnActualOff(self, input, context):
        candidates = self.activeDendrites
        for d in candidates:
            d.weaken(input, context)
            d.failureCount += 1
        self.pruneDendrites()

    def updatePredOffActualOff(self, input, context):
        for d in self.candidates:
            d.weaken(input, context)
            d.failureCount += 1
        for d in self.babyDendrites:
            if d.isCandidate(input,context):
                d.failureCount += 1
        self.pruneDendrites()
        self.pruneBabyDendrites()
        return True

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


    def findCandidateDendrites(self, input, context):
        candidates = {aDend for aDend in self.dendrites if aDend.isCandidate(input,context)}
        return candidates

    def updateOldDendrites(self, dendrites, input, context):
        for d in dendrites:
            d.strengthen(input, context)
            d.successCount += 1

    def addBabyDendrite(self, input, context):
        if len(self.babyDendrites) > MAX_NUM_BABYDENDRITES:
            self.removeWeakestBabyDendrite()
        aDend = self.makeDendrite(input, context)
        aDend.successCount = 1
        self.babyDendrites.append(aDend)
        self.countAddBabyDendrite += 1

    def makeDendrite(self, input, context):
        numInputSynapse = random.randint(int(len(input) / 2), len(input))
        numContextSynapse = random.randint(int(len(context) / 2), len(context))
        inputSynapses = set(random.sample(input, numInputSynapse))
        contextSynapses = set(random.sample(context, numContextSynapse))
        return Dendrite(inp=inputSynapses, context=contextSynapses)

    def addDendrite(self, aDend):
        if len(self.dendrites) > MAX_NUM_DENDRITES:
            self.removeWeakestDendrite()
        self.dendrites.append(aDend)
        self.countAddDendrite += 1

    def checkBabies(self, input, context):
        candidates = []
        addIndexes = []
        for i, aDend in enumerate(self.babyDendrites):
            if aDend.isCandidate(input, context):
                aDend.successCount += 1
                if aDend.predict(input, context):
                    if aDend.successCount - aDend.failureCount > 0:
                        self.addDendrite(aDend)
                        addIndexes.append(i)
                else:
                    candidates.append(aDend)
            # else:
            #     aDend.decay(input, context)
        if len(candidates) > 0:
            self.updateOldDendrites(candidates, input, context)
        else:
            self.addBabyDendrite(input, context)
        self.removeDendrites(self.babyDendrites, addIndexes)
        self.pruneBabyDendrites()

    def removeDendrites(self, dendrites, indexes):
        sortedIndexes = sorted(indexes, reverse=True)
        for i in sortedIndexes:
            dendrites.pop(i)
            
            

    def removeWeakestBabyDendrite(self):
        values = [d.sumPermanence() for d in self.babyDendrites]
        index_min = min(range(len(values)), key=values.__getitem__)
        self.babyDendrites.pop(index_min)
        self.countPruneBabyDendrite += 1

    def removeWeakestDendrite(self):
        values = [d.sumPermanence() for d in self.dendrites]
        index_min = min(range(len(values)), key=values.__getitem__)
        self.dendrites.pop(index_min)
        self.countPruneDendrite += 1

    def pruneDendrites(self):
        for i in reversed(range(len(self.dendrites))):
            aDend = self.dendrites[i]
            if aDend.failureCount - aDend.successCount > 0:
                self.dendrites.pop(i)
                self.countPruneDendrite += 1

    def pruneBabyDendrites(self):
        for i in reversed(range(len(self.babyDendrites))):
            aBaby = self.babyDendrites[i]
            if aBaby.failureCount - aBaby.successCount > 0:
                self.babyDendrites.pop(i)
                self.countPruneBabyDendrite += 1

    def resetCount(self):
        self.countPruneDendrite = 0
        self.countAddDendrite = 0
        for d in self.dendrites:
            d.resetCount()





