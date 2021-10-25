PRINT_INTERVAL = 100

import numpy as np
from models.Cell import Cell
import time
import random
random.seed(42)


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
        self.precisions = [0 for i in range(PRINT_INTERVAL)]
        self.activations = [0 for i in range(PRINT_INTERVAL)]

        self.iteration = 0

        self.startTime = time.time()
        self.programStartTime = time.time()

    def feedSparse(self, feedback={}, writer=None):
        buffer = []
        for i in range(4):
            self.actual = self.lower.feedSparse(feedback=self.pred, writer=writer)
            self.context = self.context
            # self.context = self.context | feedback
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

    # def predictMP(self, input, context):
    #     pred = Parallel(n_jobs=10)(delayed(self.predictCell)(c) for c in self.cells)
    #     # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10)
    #
    #     output = {i for i, p in enumerate(pred) if p}
    #
    #     return output
    #
    # def predictCell(self, cell):
    #     return cell.predict(self.prevActual, self.context)

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
        self.precisions[self.iteration % PRINT_INTERVAL] = \
            self.getPrecisionError(target, pred)
        self.activations[self.iteration % PRINT_INTERVAL] = len(pred)

        if self.iteration % PRINT_INTERVAL == 0:
            meanRecall = np.mean(self.recalls)
            meanPrecision = np.mean(self.precisions)
            meanActivations = np.mean(self.activations)
            currentTestTime = time.time()
            trainTime = int(currentTestTime - self.startTime)
            totalTime = int((currentTestTime - self.programStartTime)/3600)
            self.startTime = currentTestTime
            numDendrites = np.array([c.countDendrites() for c in self.cells])
            numBabyDendrites = np.array([len(c.babyDendrites) for c in self.cells])
            successCountDendrites = np.array([c.getSuccessCountDendrites() for c in self.cells])
            failureCountDendrites = np.array([c.getFailureCountDendrites() for c in self.cells])
            addBabyDendrites = np.array([c.countAddBabyDendrite for c in self.cells])
            countAddBabyDendrites = np.mean(addBabyDendrites)
            pruneBabyDendrites = np.array([c.countPruneBabyDendrite for c in self.cells])
            countPruneBabyDendrites = np.mean(pruneBabyDendrites)
            addDendrites = np.array([c.countAddDendrite for c in self.cells])
            countAddDendrites = np.mean(addDendrites)
            pruneDendrites = np.array([c.countPruneDendrite for c in self.cells])
            countPruneDendrites = np.mean(pruneDendrites)
            meanNumBabyDendrites = np.mean(numBabyDendrites)
            meanSuccessCountDendrites = np.mean(successCountDendrites)
            meanFailureCountDendrites = np.mean(failureCountDendrites)
            meanNumDendrites = np.mean(numDendrites)
            stdNumDendrites = np.std(numDendrites)
            for c in self.cells:
                c.resetCount()

            print(self.name, \
                  " Iteration: ", self.iteration,
                  " Recall: ",  "{:.4f}".format(meanRecall),
                  " Precision: ",  "{:.4f}".format(meanPrecision),
                  " muDendrites: ", "{:.1f}".format(meanNumDendrites),
                  " stdDendrites: ", "{:.1f}".format(stdNumDendrites),
                  " muAct: ", "{:.1f}".format(meanActivations),
                  " numBaby: ", "{:.1f}".format(meanNumBabyDendrites),
                  " muSucc: ", "{:.1f}".format(meanSuccessCountDendrites),
                  " muFail: ", "{:.1f}".format(meanFailureCountDendrites),
                  " addBaby: ", "{:.1f}".format(countAddBabyDendrites),
                  " pruneBaby: ", "{:.1f}".format(countPruneBabyDendrites),
                  " addDend: ", "{:.1f}".format(countAddDendrites),
                  " pruneDend: ", "{:.1f}".format(countPruneDendrites),
                  " Training Time: ", trainTime,
                  " Total Time: ", totalTime)
            writer.add_scalar('recall/recall'+self.name, meanRecall, self.iteration)
            writer.add_scalar('precision/precision'+self.name, meanPrecision, self.iteration)
            writer.add_scalar('counts/meanDendrites'+self.name, meanNumDendrites, self.iteration)
            writer.add_scalar('counts/stdDendrites'+self.name, stdNumDendrites , self.iteration)
            writer.add_scalar('counts/meanActivations'+self.name, meanActivations, self.iteration)
            writer.add_scalar('counts/numBaby'+self.name, meanNumBabyDendrites, self.iteration)
            writer.add_scalar('counts/muSucc'+self.name, meanSuccessCountDendrites, self.iteration)
            writer.add_scalar('counts/muFail'+self.name, meanFailureCountDendrites, self.iteration)
            writer.add_scalar('counts/addDend'+self.name, countAddDendrites, self.iteration)
            writer.add_scalar('counts/pruneDend'+self.name, countPruneDendrites, self.iteration)
            writer.add_scalar('counts/addBaby'+self.name, countAddBabyDendrites, self.iteration)
            writer.add_scalar('counts/pruneBaby'+self.name, countPruneBabyDendrites, self.iteration)
            writer.add_histogram('hist/numDend' + self.name, numDendrites, self.iteration)
            writer.add_histogram('hist/numSucc' + self.name, successCountDendrites, self.iteration)
            writer.add_histogram('hist/numFail' + self.name, failureCountDendrites, self.iteration)
            writer.add_histogram('hist/numAdd' + self.name, addDendrites , self.iteration)
            writer.add_histogram('hist/numPrune' + self.name, pruneDendrites , self.iteration)


    @staticmethod
    def getRecallError(target, pred):
        intersection = target & pred
        recall = len(intersection) / (len(target) + 0.0001)
        # if recall > 0.99:
        #     print("Hello ", self.name)
        return recall
    @staticmethod
    def getPrecisionError(target, pred):
        intersection = target & pred
        recall = len(intersection) / (len(pred) + 0.0001)
        # if recall > 0.99:
        #     print("Hello ", self.name)
        return recall
