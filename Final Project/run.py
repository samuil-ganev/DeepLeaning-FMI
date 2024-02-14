#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *
import bpe

startToken = '<S>'
endToken = '</S>'
unkToken = '<UNK>'
padToken = '<PAD>'

class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor

    def get_lr(self):
        step_num = self.last_epoch + 1
        lr = 1 * (512 ** -0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
        print(lr)
        return [lr for _ in self.optimizer.param_groups]

def perplexity(nmt, sourceTest, targetTest, batchSize):
    testSize = len(sourceTest)
    H = 0.
    c = 0
    for b in range(0,testSize,batchSize):
        sourceBatch = sourceTest[b:min(b+batchSize, testSize)]
        targetBatch = targetTest[b:min(b+batchSize, testSize)]
        l = sum(len(s)-1 for s in targetBatch)
        c += l
        with torch.no_grad():
            H += l * nmt(sourceBatch,targetBatch)
    return math.exp(H/c)

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    sourceCorpus, sourceWord2ind, targetCorpus, targetWord2ind, sourceDev, targetDev = utils.prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken)

    # Думи на английски: 20299, Думи на български: 37455

    # pickle.dump((sourceCorpus, targetCorpus), open(corpusOriginalFileName, 'wb'))

    pickle.dump((sourceCorpus,targetCorpus,sourceDev,targetDev), open(corpusDataFileName, 'wb'))
    pickle.dump((sourceWord2ind,targetWord2ind), open(wordsDataFileName, 'wb'))

    # sourcePath, sourceCorpus, sourcePair2ind = bpe.bpe(sourceCorpus, byte_pairs_eng, 'eng', bpe_dropout)
    # targetPath, targetCorpus, targetPair2ind = bpe.bpe(targetCorpus, byte_pairs_bg, 'bg', bpe_dropout)
    # sourceDev, targetDev = bpe.encode_corpus(sourceDev, sourcePath, 'eng', 0), bpe.encode_corpus(targetDev, targetPath, 'bg', 0)

    # pickle.dump((sourceCorpus, targetCorpus, sourceDev, targetDev), open(corpusDataFileName, 'wb'))
    # pickle.dump((sourcePair2ind, targetPair2ind), open(wordsDataFileName, 'wb'))
    # pickle.dump((sourcePath, targetPath), open(pathsFileName, 'wb'))

    print('Data prepared.')

if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    # (sourceCorpusOriginal, targetCorpusOriginal) = pickle.load(open(corpusOriginalFileName, 'rb'))
    (sourceCorpus, targetCorpus, sourceDev, targetDev) = pickle.load(open(corpusDataFileName, 'rb'))
    (sourcePair2ind, targetPair2ind) = pickle.load(open(wordsDataFileName, 'rb'))
    (sourcePath, targetPath) = pickle.load(open(pathsFileName, 'rb'))

    print(sourceCorpus[0:5])
    print(targetCorpus[0:5])
    print(len(sourcePair2ind), len(targetPair2ind))

    nmt = model.NMTmodel(sourcePair2ind, targetPair2ind, sourcePath, targetPath, unkToken, padToken, num_layers, n_head, d_model, d_keys, d_values, d_ff, dropout, temperature).to(device)
    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=0.000000001)

    lr_scheduler = NoamScheduler(optimizer, d_model, warmup_steps)

    if sys.argv[1] == 'extratrain':
        # sourceCorpus, targetCorpus = bpe.encode_corpus(sourceCorpusOriginal, sourcePath, 'eng', bpe_dropout), bpe.encode_corpus(targetCorpusOriginal, targetPath, 'bg', bpe_dropout)
        nmt.load(modelFileName)
        (bestPerplexity,learning_rate,osd,lr) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        lr_scheduler.load_state_dict(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf

    idx = np.arange(len(sourceCorpus), dtype='int32')
    nmt.train()
    trial = 0
    patience = 0
    iter = 0
    beginTime = time.time()
    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        targetWords = 0
        trainTime = time.time()
        for b in range(0, len(idx), batchSize):
            iter += 1
            sourceBatch = [ sourceCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            targetBatch = [ targetCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]

            st = sorted(list(zip(sourceBatch,targetBatch)),key=lambda e: len(e[0]), reverse=True)
            (sourceBatch,targetBatch) = tuple(zip(*st))
            targetWords += sum( len(s)-1 for s in targetBatch )
            H = nmt(sourceBatch, targetBatch)
            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            lr_scheduler.step()
            if iter % log_every == 0:
                print("Iteration:",iter,"Epoch:",epoch+1,'/',maxEpochs,", Batch:",b//batchSize+1, '/', len(idx) // batchSize+1, ", loss: ",H.item(), "words/sec:",targetWords / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                targetWords = 0
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, sourceDev, targetDev, batchSize)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    patience = 0
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((bestPerplexity,learning_rate,optimizer.state_dict(),lr_scheduler.state_dict()), modelFileName + '.optim')
                else:
                    patience += 1
                    if patience == max_patience:

                        trial += 1
                        # if trial == max_trials:
                        #     print('early stop!')
                        #     exit(0)
                        # learning_rate *= learning_rate_decay
                        # print('load previously best model and decay learning rate to:', learning_rate)
                        # nmt.load(modelFileName)
                        # (bestPerplexity,_,osd) = torch.load(modelFileName + '.optim')
                        # optimizer.load_state_dict(osd)
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = learning_rate
                        patience = 0

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, sourceDev, targetDev, batchSize)
    print('Last model perplexity: ',currentPerplexity)

    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((bestPerplexity,learning_rate,optimizer.state_dict(), lr_scheduler.state_dict()), modelFileName + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    (sourcePair2ind,targetPair2ind) = pickle.load(open(wordsDataFileName, 'rb'))
    (sourcePath, targetPath) = pickle.load(open(pathsFileName, 'rb'))

    nmt = model.NMTmodel(sourcePair2ind, targetPair2ind, sourcePath, targetPath, unkToken, padToken, num_layers, n_head, d_model, d_keys, d_values, d_ff, dropout, temperature).to(device)
    nmt.load(modelFileName)

    # sourceTest = pickle.load(open(sourceTestFileName, 'rb'))
    # targetTest = pickle.load(open(targetTestFileName, 'rb'))

    (sourceTest, targetTest) = bpe.encode_corpus(sys.argv[2], sourcePath, 'eng', 0), bpe.encode_corpus(sys.argv[3], targetPath, 'bg', 0)

    # sourceTest = utils.readCorpus(sys.argv[2])
    # targetTest = utils.readCorpus(sys.argv[3])
    # targetTest = [ [startToken] + s + [endToken] for s in targetTest]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, sourceTest, targetTest, batchSize))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    (sourcePair2ind,targetPair2ind) = pickle.load(open(wordsDataFileName, 'rb'))
    (sourcePath, targetPath) = pickle.load(open(pathsFileName, 'rb'))

    sourceTest = utils.readCorpus(sys.argv[2])

    nmt = model.NMTmodel(sourcePair2ind, targetPair2ind, sourcePath, targetPath, unkToken, padToken, num_layers, n_head, d_model, d_keys, d_values, d_ff, dropout, temperature).to(device)
    nmt.load(modelFileName)
    nmt.to(device)

    nmt.eval()
    file = open(sys.argv[3],'w', encoding='utf-8')
    pb = utils.progressBar()
    pb.start(len(sourceTest))
    for s in sourceTest:
        file.write(' '.join(nmt.translateSentence(s))+"\n")
        pb.tick()
    pb.stop()

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
