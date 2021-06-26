import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import sys

np.random.seed(123)


def LoadData(filename):
    fileData = open(filename, 'r', encoding='utf8').read()
    allChars = list(set(fileData))

    # To make list deterministic
    allChars.sort()
    random.Random(4).shuffle(allChars)

    charToInd = OrderedDict(
        (char, ind) for ind, char in enumerate(allChars))

    indToChar = OrderedDict((ind, char) for ind, char in
                            enumerate(allChars))

    data = {"fileData": fileData, "allChars": allChars,
            "nVocab": len(allChars), "charToInd": charToInd,
            "indToChar": indToChar}

    return data


class RNN():

    def __init__(self, data, m=100, eta=0.1, seqLength=25):

        self.m = m
        self.N = seqLength
        self.eta = eta

        for key, value in data.items():
            setattr(self, key, value)

        self.b, self.c, self.U, self.V, self.W = \
            self.InitParameters(self.m, self.nVocab)

    def ComputeSoftmax(self, x):
        return np.exp(x - np.max(x, axis=0)) / \
            np.exp(x - np.max(x, axis=0)).sum(axis=0)

    def ComputeTanh(self, x):
        return np.tanh(x)

    def InitParameters(self, m, K, sig=0.01):
        b = np.zeros((m, 1))
        c = np.zeros((K, 1))

        U = np.random.normal(0, sig, size=(m, K))
        W = np.random.normal(0, sig, size=(m, m))
        V = np.random.normal(0, sig, size=(K, m))

        return b, c, U, V, W

    def EvaluateRNN(self, hiddenState, x):
        a = self.W@hiddenState + self.U@x + self.b
        hiddenState = self.ComputeTanh(a)
        o = self.V@hiddenState + self.c
        probability = self.ComputeSoftmax(o)

        return a, hiddenState, o, probability

    def Computegrads(self, X, Y, hPrevious):
        aDict = dict()
        xDict = dict()
        hDict = dict()
        oDict = dict()
        pDict = dict()

        hDict[-1] = np.copy(hPrevious)

        loss = 0

        for n in range(len(X)):
            xDict[n] = np.zeros((self.nVocab, 1))
            xDict[n][X[n]] = 1

            aDict[n], hDict[n], oDict[n], pDict[n] = self.EvaluateRNN(
                hDict[n-1], xDict[n])

            pLoss = pDict[n][Y[n]][0]
            loss = loss + -np.log(pLoss)

        grads = dict()
        grads['W'] = np.zeros_like(self.W)
        grads['U'] = np.zeros_like(self.U)
        grads['V'] = np.zeros_like(self.V)
        grads['b'] = np.zeros_like(self.b)
        grads['c'] = np.zeros_like(self.c)

        grads['o'] = np.zeros_like(pDict[0])
        grads['h'] = np.zeros_like(hDict[0])
        grads['hNext'] = np.zeros_like(hDict[0])
        grads['a'] = np.zeros_like(aDict[0])

        for n in reversed(range(len(X))):
            grads["o"] = np.copy(pDict[n])
            grads["o"][Y[n]] -= 1

            grads["V"] = grads["V"] + grads["o"]@hDict[n].T
            grads["c"] = grads["c"] + grads["o"]

            grads["h"] = self.V.T@grads["o"] + grads["hNext"]

            grads["a"] = np.multiply(
                grads["h"], (1 - np.square(hDict[n])))

            grads["hNext"] = self.W.T@grads["a"]

            grads["U"] = grads["U"] + grads["a"]@xDict[n].T
            grads["W"] = grads["W"] + grads["a"]@hDict[n-1].T
            grads["b"] = grads["b"] + grads["a"]

        labelsToKeep = ["a", "o", "h", "hNext"]
        for label in grads:
            if label not in labelsToKeep:
                grads[label] = grads[label]

        for gradient in grads:
            grads[gradient] = np.clip(grads[gradient], -5, 5)

        return grads, loss, hDict[len(X)-1]

    def SynthesizeTextSequence(self, hiddenState, index, seqLength, textBuffer=""):
        x = np.zeros((self.nVocab, 1))
        x[index] = 1

        for _ in range(seqLength):
            a, hiddenState, o, probability = self.EvaluateRNN(hiddenState, x)
            index = np.random.choice(range(self.nVocab), p=probability.flat)
            x = np.zeros((self.nVocab, 1))
            x[index] = 1
            textBuffer += self.indToChar[index]

        return textBuffer


def GenerateAccuracyGraphs(smoothLosses, n_epochs, label="", yLimit=None):

    fig, ax = plt.subplots(figsize=(7, 5))
    # ax.plot(np.arange(n_epochs), losses, label="Loss")
    ax.plot(smoothLosses, label="Smooth Loss")
    ax.legend()

    # if yLimit and (maxY < yLimit):
    # ax.set_ylim([0, yLimit])

    ax.set(xlabel='Update Steps (x1000)', ylabel=label)
    # plt.figtext(0.5, 0.92, 'n_epochs:' + str(n_epochs), ha='center',
    #            bbox={"facecolor": "blue", "alpha": 0.2, "pad": 5})
    ax.grid()

    plt.savefig('output/' + label + 'Graph' +
                'Epochs' + str(n_epochs) + '.png')


def trainRoutine(rnn, params, memParams, n_epochs=3, pos=0, i=0, EarlyStopIteration=-1, NCharsToPrint=20):
    smoothLosses = []
    NmbEpoch = 0
    endOfBook = len(rnn.fileData) - rnn.N - 1

    while NmbEpoch < n_epochs:
        if i < 1 or pos >= endOfBook:
            pos, hPrevious = 0, np.zeros((rnn.m, 1))
            NmbEpoch += 1

        X, Y = [], []
        for char in rnn.fileData[pos:pos+rnn.N]:
            X.append(rnn.charToInd[char])

        for char in rnn.fileData[pos+1:pos+rnn.N+1]:
            Y.append(rnn.charToInd[char])

        grads, loss, hPrevious = rnn.Computegrads(X, Y, hPrevious)

        if i < 1 and NmbEpoch == 1:
            smoothLoss = loss
        smoothLoss = 0.999 * smoothLoss + 0.001 * loss

        if i % 1000 == 0:
            smoothLosses.append(smoothLoss)
            txt = rnn.SynthesizeTextSequence(hPrevious, X[0], 200)
            print('Update Step:' + str(i) + ', Smooth Loss:', smoothLoss)
            print('Synthezised Text:')
            print()
            print(txt[:NCharsToPrint])
            print()

        if i % 100 == 0:
            print('Update Step:', str(i) + ', Smooth Loss:', smoothLoss)

        for key in params:
            memParams[key] += grads[key] * grads[key]
            params[key] -= rnn.eta / np.sqrt(memParams[key] +
                                             np.finfo(float).eps) * grads[key]

        pos += rnn.N
        i += 1

        if NmbEpoch == n_epochs:
            print('Done with Training with', smoothLosses[-1],
                  'smooth loss after', i, 'update steps')
            GenerateAccuracyGraphs(
                smoothLosses, n_epochs=n_epochs, label="SmoothLoss", yLimit=None)
            txt = rnn.SynthesizeTextSequence(hPrevious, X[0], 1000)
            print('Synthezised Text:')
            print()
            print(txt)
            print()
            break

        if i == EarlyStopIteration:
            break


datasetPath = "../../Datasets/goblet_book.txt"
data = LoadData(datasetPath)

# 43001 steps per epoch
n_epochs = 5

rnn = RNN(data)
params = {"W": rnn.W, "U": rnn.U, "V": rnn.V, "b": rnn.b, "c": rnn.c}

memParams = {"W": np.zeros_like(rnn.W), "U": np.zeros_like(rnn.U),
             "V": np.zeros_like(rnn.V), "b": np.zeros_like(rnn.b),
             "c": np.zeros_like(rnn.c)}

# EarlyStopIteration = -1 completes full training routine
trainRoutine(rnn, params, memParams,
             n_epochs=n_epochs,
             EarlyStopIteration=-1,
             NCharsToPrint=200)
