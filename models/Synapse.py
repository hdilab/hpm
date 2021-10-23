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
        if self.permanence > CONNECTION_THRESHOLD:
            self.isActive = True
        else:
            self.isActive = False


    def strengthen(self):
        self.permanence += POSITIVE_REINFORCE
        self.updateConnection()

    def weaken(self):
        self.permanence -= NEGATIVE_REINFORCE
        self.updateConnection()

    def updateConnection(self):
        if self.permanence > CONNECTION_THRESHOLD:
            self.isActive = True
        else:
            self.isActive = False
