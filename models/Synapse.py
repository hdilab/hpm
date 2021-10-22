CONNECTION_THRESHOLD = 0.5
POSITIVE_REINFORCE = 0.1
NEGATIVE_REINFORCE = 0.1


class Synapse(object):
    def __init__(self,
                 target=None,
                 permanence=0.8
                 ):

        super().__init__()

        self.target = target
        self.permanence = permanence

    def isActive(self):
        if self.permanence > CONNECTION_THRESHOLD:
            return True
        else:
            return False

    def strengthen(self):
        self.permanence += POSITIVE_REINFORCE

    def weaken(self):
        self.permanence -= NEGATIVE_REINFORCE
