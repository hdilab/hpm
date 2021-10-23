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

    def predict(self, input, context):
        for aDend in self.dendrites:
            if aDend.predict(input, context):
                return True
        return False

    def updatePredOnActualOn(self, input, context):
        candidates = self.findActiveDendrites(input, context)
        self.updateOldDendrites(candidates, input, context)

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
        self.pruneDendrites()

    def findActiveDendrites(self, input, context):
        activeDendrites = []
        for d in self.dendrites:
            if d.predict(input, context):
                activeDendrites.append(d)
        return activeDendrites

    def countDendrites(self):
        return len(self.dendrites)

    def updatePredOffActualOff(self, input, context):
        for d in self.dendrites:
            d.decay()

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

    def removeWeakestDendrite(self):
        values = [d.sumPermanence() for d in self.dendrites]
        index_min = min(range(len(values)), key=values.__getitem__)
        self.dendrites.pop(index_min)

    def pruneDendrites(self):
        for i  in reversed(range(len(self.dendrites))):
            if self.dendrites[i].hasNegativePermanence():
                self.dendrites.pop(i)





