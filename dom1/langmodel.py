import sys
import math


class progressBar:
    def __init__(self, barWidth=50):
        self.barWidth = barWidth
        self.period = None

    def start(self, count):
        self.item = 0
        self.period = int(count / self.barWidth)
        sys.stdout.write("[" + (" " * self.barWidth) + "]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth + 1))

    def tick(self):
        if self.item > 0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1

    def stop(self):
        sys.stdout.write("]\n")


alphabet = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'ю', 'я', ' ', '-']
startToken = '<START>'
endToken = '<END>'
unkToken = '<UNK>'


class MarkovModel:

    def __init__(self, corpus, K, dictionaryLimit=10000, startToken='<START>', endToken='<END>', unkToken='<UNK>'):
        self.K = K
        self.startToken = startToken
        self.endToken = endToken
        self.unkToken = unkToken
        self.kgrams = {}
        self.extractMonograms(corpus, dictionaryLimit)
        for k in range(2, K + 1):
            self.extractKgrams(corpus, k)
        self.Tc = {}
        for context in self.kgrams:
            self.Tc[context] = sum(self.kgrams[context][v] for v in self.kgrams[context])

    def extractMonograms(self, corpus, limit):
        pb = progressBar()
        pb.start(len(corpus))
        dictionary = {}
        for sent in corpus:
            pb.tick()
            for w in sent[1:]:
                if w not in dictionary:
                    dictionary[w] = 0
                dictionary[w] += 1
        L = sorted([(w, dictionary[w]) for w in dictionary], key=lambda x: x[1], reverse=True)
        if limit > len(L): limit = len(L)
        mono = {w: c for (w, c) in L[:limit]}
        sumUnk = sum(c for (w, c) in L[limit:])
        mono[self.unkToken] = sumUnk
        self.kgrams[tuple()] = mono
        pb.stop()

    def substituteUnkownWords(self, sentence):
        return [sentence[0]] + [w if w in self.kgrams[tuple()] else self.unkToken for w in sentence[1:]]

    def extractKgrams(self, corpus, k):
        pb = progressBar()
        pb.start(len(corpus))
        for s in corpus:
            pb.tick()
            sent = self.substituteUnkownWords(s)
            for i in range(k - 1, len(sent)):
                w = sent[i]
                context = tuple(sent[i - k + 1:i])
                if context not in self.kgrams: self.kgrams[context] = {}
                if w not in self.kgrams[context]: self.kgrams[context][w] = 0
                self.kgrams[context][w] += 1
        pb.stop()

    def probMLE(self, w, context):
        if context not in self.kgrams:
            return 0.0
        elif w not in self.kgrams[context]:
            return 0.0
        else:
            return self.kgrams[context][w] / self.Tc[context]

    def prob(self, w, context, alpha):
        if len(context) > 0:
            return alpha * self.probMLE(w, context) + (1 - alpha) * self.prob(w, context[1:], alpha)
        else:
            return self.probMLE(w, context)

    def sentenceLogProbability(self, s, alpha):
        sent = self.substituteUnkownWords(s)
        return sum(
            math.log(self.prob(sent[i], tuple(sent[max(0, i - self.K + 1):i]), alpha), 2) for i in range(1, len(sent)))

    def bestContinuation(self, sentence, alpha, n):
        l = len(sentence)
        sentence = self.substituteUnkownWords(sentence)
        context = tuple(sentence[max(0, l - self.K + 1):l])
        while not context in self.kgrams:
            context = context[1:]
        L = [(w, self.prob(w, context, alpha)) for w in self.kgrams[context]]
        return sorted(L, key=lambda x: x[1], reverse=True)[:n]

    def perplexity(self, corpus, alpha):
        pb = progressBar()
        pb.start(len(corpus))
        m = sum(len(s) - 1 for s in corpus)
        crossEntropy = 0.0
        for s in corpus:
            pb.tick()
            crossEntropy -= self.sentenceLogProbability(s, alpha)
        crossEntropyRate = crossEntropy / m
        pb.stop()
        return 2 ** crossEntropyRate

