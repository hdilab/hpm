import random

from models.Synapse import Synapse

MAX_SYNAPSES = 40
MIN_SYNAPSES = 20
ACTIVATE_THRESHOLD = 10


class Dendrite(object):
    def __init__(self,
                 inp=None,
                 context=None):
        super().__init__()

        self.inputSynapses = self.populateSynapses(inp)
        self.contextSynapses = self.populateSynapses(context)
        self.predictionCount = 0
        self.successCount = 0
        self.failureCount = 0

    def predict(self, inp, context):
        self.predictionCount += 1
        matchInput = [1 for s in self.inputSynapses if s.isActive and s.target in inp]
        if sum(matchInput) < ACTIVATE_THRESHOLD:
            return False

        matchContext = [1 for s in self.contextSynapses if s.isActive and s.target in context]
        if sum(matchContext) < ACTIVATE_THRESHOLD:
            return False
        else:
            return True

    def isCandidate(self, inp, context):
        matchInput = {s for s in self.inputSynapses if s.target in inp}
        matchContext = {s for s in self.contextSynapses if s.target in context}
        if len(matchInput) > ACTIVATE_THRESHOLD and len(matchContext) > ACTIVATE_THRESHOLD:
            return True
        else:
            return False

    def strengthen(self, inp, context):
        for s in self.inputSynapses:
            if s.target in inp:
                s.strengthen()
        for s in self.contextSynapses:
            if s.target in context:
                s.strengthen()

    def weaken(self, inp, context):
        for s in self.inputSynapses:
            if s.target in inp:
                s.weaken()
        for s in self.contextSynapses:
            if s.target in context:
                s.weaken()

    def decay(self):
        for s in self.inputSynapses:
            s.decay()
        for s in self.contextSynapses:
            s.decay()

    @staticmethod
    def populateSynapses(indexes):
        synapses = [Synapse(permanence=random.uniform(0.5,1.0), target=i) for i in indexes]
        return synapses

    def sumPermanence(self):
        inputPermanences = [s.permanence for s in self.inputSynapses]
        contextPermanences = [s.permanence for s in self.contextSynapses]
        sumPermanences = sum(inputPermanences) + sum(contextPermanences)
        return sumPermanences

    def hasNegativePermanence(self):
        minInputPermanences = min([s.permanence for s in self.inputSynapses])
        minContextPermanences = min([s.permanence for s in self.contextSynapses])
        if minInputPermanences < 0 or minContextPermanences < 0:
            return True
        else:
            return False
