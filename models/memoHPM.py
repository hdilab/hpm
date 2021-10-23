PRINT_INTERVAL = 100

import numpy as np
from models.Cell import Cell
import time
import random
random.seed(42)
from joblib import Parallel, delayed


class memoHPM(object):
    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 lower=None,
                 name="layer"):

        super().__init__()
        self.cells = [Cell(index=i) for i in range(numBits)]

        self.lower = lower
        self.name = name
        self.numBits = numBits
        self.numOnBits = numOnBits

        self.population = [i for i in range(numBits)]

        self.prevActual = set(random.sample(self.population, numOnBits))
        self.actual = set(random.sample(self.population, numOnBits))
        self.pred = set(random.sample(self.population, numOnBits))
        self.context = set(random.sample(self.population, numOnBits))

        self.recalls = [0 for i in range(PRINT_INTERVAL)]
        self.iteration = 0

        self.startTime = time.time()
        self.programStartTime = time.time()

    def feedSparse(self, feedback=None, writer=None):
        buffer = []
        for i in range(4):
            self.actual = self.lower.feedSparse(feedback=self.pred, writer=writer)
            self.pred = self.predict(self.prevActual, self.context)
            self.evaluate(self.pred, self.actual, writer)
            self.update(self.prevActual, self.context, writer=writer)
            buffer.append(self.actual)
            self.prevActual = self.actual
            self.iteration += 1
        poolOutput = self.pool(buffer, writer)
        self.context = poolOutput
        return self.context

    def predict(self, input, context):
        pred = {i for i in range(len(self.cells)) if self.cells[i].predict(input, context)}
        return pred

    def predictMP(self, input, context):
        pred = Parallel(n_jobs=10)(delayed(self.predictCell)(c) for c in self.cells)
        # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10)

        output = {i for i, p in enumerate(pred) if p}

        return output

    def predictCell(self, cell):
        return cell.predict(self.prevActual, self.context)

    def pool(self, buffer, writer):
        combine = []
        for i, buf in enumerate(buffer):
            for cell in buf:
                combine.append(cell + i*self.numBits)
        output = {int(c/4) for c in combine if c % 4 == 0}
        return output

    def update(self, input, context, writer=None):
        for i, cell in enumerate(self.cells):
            if i in self.pred and i in self.actual:
                cell.updatePredOnActualOn(input, context)
            elif i in self.pred and i not in self.actual:
                cell.updatePredOnActualOff(input, context)
            elif i not in self.pred and i in self.actual:
                cell.updatePredOffActualOn(input, context)
            elif i not in self.pred and i not in self.actual:
                cell.updatePredOffActualOff(input, context)

    def evaluate(self, pred, target, writer):

        self.recalls[self.iteration % PRINT_INTERVAL] = \
            self.getRecallError(target, pred)

        if self.iteration % PRINT_INTERVAL == 0:
            meanRecall = np.mean(self.recalls)
            currentTestTime = time.time()
            trainTime = int(currentTestTime - self.startTime)
            totalTime = int((currentTestTime - self.programStartTime)/3600)
            self.startTime = currentTestTime

            print(self.name, \
                  "\t Iteration: \t", self.iteration,
                  "\t Recall: \t",  "{:.4f}".format(meanRecall),
                  "\t Training Time: \t", trainTime,
                  "\t Total Time: \t", totalTime)
            writer.add_scalar('recall/recall'+self.name, meanRecall, self.iteration)

    @staticmethod
    def getRecallError(target, pred):
        intersection = target & pred
        recall = len(intersection) / (len(target) + 0.0001)
        # if recall > 0.99:
        #     print("Hello ", self.name)
        return recall
