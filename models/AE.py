# Simple autoencoder
# Source code from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

from torch import nn
from models.kWTA import Sparsify1D
import torch

class autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='kWTA_AE',
                 numOnBits=10):
        super().__init__()
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.encoderL1 = nn.Linear(numBits*4, numBits*2)
        self.encoderL2 = nn.Linear(numBits*2, numBits)
        self.decoderL1 = nn.Linear(numBits, numBits*2)
        self.decoderL2 = nn.Linear(numBits*2, numBits*4)

    def forward(self, x):
        k = self.numOnBits
        L1out = self.encoderL1(x);
        tmpL1 = L1out.view(x.shape[0],-1)
        L1TopVal = tmpL1.topk(k * 2)[0][:, -1]
        L1Sparse = (L1out > L1TopVal).to(L1out)

        L2out = self.encoderL2(L1Sparse)
        L2TopVal = L2out.topk(k * 1)[0][:, -1]
        sparseEmb = (L2out > L2TopVal).to(L2out)

        decoderL1out = self.decoderL1(sparseEmb);
        decoderL1TopVal = decoderL1out.topk(k * 2)[0][:, -1]
        decoderL1Sparse = (decoderL1out > decoderL1TopVal).to(decoderL1out)

        decoderL2out = self.decoderL2(decoderL1Sparse)
        decoderL2TopVal = decoderL2out.topk(k * 4)[0][:, -1]
        out = (decoderL2out > decoderL2TopVal).to(decoderL2out)
        return out, sparseEmb

class simple_autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='simple_AE',
                 numOnBits=10):
        super().__init__()

        self.numBits = numBits
        self.numOnBits = numOnBits
        self.encoder = nn.Sequential(
            nn.Linear(numBits*4, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits))
        self.decoder = nn.Sequential(
            nn.Linear(numBits, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits*4))

    def forward(self, x):
        emb = self.encoder(x)
        x = self.decoder(emb)
        return x, emb

class simple_autoencoder2(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='kWTA_AE',
                 numOnBits=10):
        super().__init__()
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.encoderL1 = nn.Linear(numBits*4, numBits*2)
        self.encoderL2 = nn.Linear(numBits*2, numBits)
        self.decoderL1 = nn.Linear(numBits, numBits*2)
        self.decoderL2 = nn.Linear(numBits*2, numBits*4)

    def forward(self, x):
        L1out = self.encoderL1(x);
        L2input = nn.functional.relu(L1out)
        L2out = self.encoderL2(L2input)

        decoderL1out = self.decoderL1(L2out);
        decoderL2input = nn.functional.relu(decoderL1out)
        decoderL2out = self.decoderL2(decoderL2input)
        return decoderL2out, L2out


class kWTA_autoencoder(nn.Module):

    def __init__(self,
                 numBits=512,
                 name='kWTA autoencoder',
                 numOnBits=10):
        super().__init__()

        learningRate = 1e-2
        self.numBits = numBits
        self.numOnBits = numOnBits
        self.encoder = nn.Sequential(
            nn.Linear(numBits*4, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits))
        self.decoder = nn.Sequential(
            nn.Linear(numBits, numBits*2),
            nn.ReLU(True),
            nn.Linear(numBits*2, numBits*4))
        self.kWTA = Sparsify1D(numOnBits)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                     lr=learningRate,
                                     weight_decay=1e-5)

    def forward(self, x):
        out = self.encoder(x)
        emb = self.kWTA(out)
        x = self.decoder(emb)
        return x, emb

    def testBinaryEmbedding(self, x):
        out = self.encoder(x)
        emb = self.kWTA(out)
        binaryEmb = (emb != 0).to(emb)
        x = self.decoder(binaryEmb)
        return x, binaryEmb

    def pool(self, x, writer):
        recon, emb = self.forward(x)
        binaryEmb = (emb != 0).to(emb)
        loss = self.criterion(recon, x) + torch.mean((binaryEmb - emb)*(binaryEmb-emb))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.evaluate(x, binaryEmb, writer)
        # self.update()
        # self.iteration += 1
        return binaryEmb

    def evaluate(self, input, emb, writer):
        if i % TestInterval == 0:
            startTestTime = time.time()
            testLoss = 0.0;
            recall = 0.0
            topValuesHistory = torch.zeros(numTestCharacter, NumOnBits * 8)
            numTest = int(numTestCharacter / 4)
            with torch.no_grad():
                for j in range(numTest):
                    input = []
                    for _ in range(4):
                        signal = trainFeeder.feed()
                        signal = np.squeeze(signal).tolist()
                        input.extend(signal)
                    input = torch.tensor(input)
                    input = torch.reshape(input, (1, -1))
                    recon, emb = AE.testBinaryEmbedding(input)
                    topValues, topIndices = torch.topk(recon, NumOnBits * 8)
                    topValues = torch.sigmoid(topValues)
                    topValuesHistory[j, :] = topValues

                    _, topIndices = torch.topk(recon, NumOnBits * 4)
                    _, inputIndices = torch.topk(input, NumOnBits * 4)

                    # Evaluate character level
                    reconTopVal = recon.topk(NumOnBits * 4)[0][:, -1]
                    reconChars = (recon > reconTopVal).to(recon)
                    reconChars = reconChars.numpy()
                    reconChars = reconChars.reshape((4, -1))

                    inputChars = input.numpy()
                    inputChars = inputChars.reshape((4, -1))
                    for c in range(4):
                        inputChar = char_sdr.getInput(inputChars[c, :])
                        reconChar = char_sdr.getInput(reconChars[c, :])
                        if inputChar == reconChar:
                            accuracy += 1

                    if j > numTest - 10:
                        print('-----------------')
                        for c in range(4):
                            inputChar = char_sdr.getInput(inputChars[c, :])
                            reconChar = char_sdr.getInput(reconChars[c, :])
                            if inputChar == reconChar:
                                print(inputChar)
                            else:
                                print(
                                    "Not Match: " + inputChar + " " + reconChar + ' : ' + str(np.sum(reconChars[c, :])))

                    listInput = inputIndices.tolist()[0]
                    setInput = set(listInput)
                    listTopIndices = topIndices.tolist()[0]
                    setTopIndices = set(listTopIndices)
                    intersection = setInput.intersection(setTopIndices)
                    recall += len(intersection) / len(setInput)
                    loss = criterion(recon, input)
                    testLoss += loss

            testLoss /= numTest
            trainLoss /= TestInterval
            recall /= numTest
            accuracy /= numTestCharacter
            endTestTime = time.time()
            trainingTime = int(startTestTime - startTrainingTime)
            testTime = int(endTestTime - startTestTime)
            startTrainingTime = time.time()

            print(
                'epoch [{}/{}], Test Loss:{:.6f},  Train Loss:{:.6f}, Recall:{:.6f}, Accuracy:{:6f} Training Time:{} Test Time: {}'
                .format(i + 1, n_epochs, testLoss, trainLoss, recall, accuracy, trainingTime, testTime))
            writer.add_scalar('test/AE-BCE', testLoss, i)
            writer.add_scalar('train/AE-BCE', trainLoss, i)
            writer.add_scalar('test/AE-Recall', recall, i)
            writer.add_scalar('test/AE-Accuracy', accuracy, i)
            trainLoss = 0.0
            writer.add_histogram('AE.decoder.linear2.weight', AE.decoder[2].weight, i)
            writer.add_histogram('AE.decoder.linear2.bias', AE.decoder[2].bias, i)
            writer.add_histogram('AE.output', recon, i)
            writer.add_histogram('AE.input', input, i)
            writer.add_histogram('AE.embedding', emb, i)
            writer.add_histogram('AE.output.TopValues', topValuesHistory, i)