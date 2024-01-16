#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch
import math

def trainModel(trainCorpus, lm, optimizer, epochs, batchSize):
    idx = np.arange(len(trainCorpus), dtype='int32')
    lm.train()
    for epoch in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), batchSize):
            batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            H = lm(batch)
            optimizer.zero_grad()
            H.backward()
            optimizer.step()
            print("Epoch:",epoch,'/',epochs,", Batch:",b // batchSize, '/', len(idx) // batchSize, ", loss: ",H.item())

def perplexity(lm, testCorpus, batchSize):
    lm.eval()
    H = 0.
    c = 0
    for b in range(0,len(testCorpus),batchSize):
        batch = testCorpus[b:min(b+batchSize, len(testCorpus))]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * lm(batch)
    return math.exp(H/c)
