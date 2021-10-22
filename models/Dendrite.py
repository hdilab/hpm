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

    def predict(self, inp, context):
        matchInput = {s for s in self.inputSynapses if s.isActive() and s.target in inp}
        matchContext = {s for s in self.contextSynapses if s.isActive() and s.target in context}
        if len(matchInput) > ACTIVATE_THRESHOLD and len(matchContext) > ACTIVATE_THRESHOLD:
            return True
        else:
            return False

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


    @staticmethod
    def populateSynapses(indexes):
        synapses = [Synapse(permanence=random.uniform(0.5,1.0), target=i) for i in indexes]
        return synapses

    def sumPermanence(self):
        inputPermanences = [s.permanence for s in self.inputSynapses]
        contextPermanences = [s.permanence for s in self.contextSynapses]
        sumPermanences = sum(inputPermanences) + sum(contextPermanences)
        return sumPermanences
