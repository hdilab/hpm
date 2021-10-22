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
        self.activeDendrites = []
        activate = False
        for aDend in self.dendrites:
            if aDend.predict(input, context):
                activate = True
                self.activeDendrites.append(aDend)
        return activate

    def updatePredOnActualOn(self, input, context):
        self.updateOldDendrites(self.activeDendrites, input, context)
        return True

    def updatePredOffActualOn(self, input, context):
        candidates = self.findCandidateDendrites(input, context)
        if len(candidates) > 0:
            self.updateOldDendrites(candidates, input, context)
        else:
            self.addDendrite(input, context)

    def updatePredOnActualOff(self, input, context):
        for d in self.activeDendrites:
            d.weaken(input, context)
        return True

    def updatePredOffActualOff(self, input, context):
        return True

    def findCandidateDendrites(self, input, context):
        candidates = {aDend for aDend in self.dendrites if aDend.isCandidate(input,context)}
        return candidates

    def updateOldDendrites(self, dendrites, input, context):
        for d in dendrites:
            d.strengthen(input, context)

    def addDendrite(self, input, context):
        if len(self.dendrites) > MAX_NUM_DENDRITES:
            self.pruneDendrite()
        self.dendrites.append(Dendrite(inp=input, context=context))

    def pruneDendrite(self):
        values = [d.sumPermanence() for d in self.dendrites]
        index_min = min(range(len(values)), key=values.__getitem__)
        self.dendrites.pop(index_min)



