#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
##########################################################################
###
### Домашно задание 3
###
#############################################################################

import random

corpusSplitString = ';)\n'
maxProgramLength = 10000
symbolCountThreshold = 100

def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus

def getAlphabet(corpus):
    symbols={}
    for s in corpus:
        for c in s:
            if c in symbols: symbols[c] += 1
            else: symbols[c]=1
    return symbols

def prepareData(corpusFileName, startChar, endChar, unkChar, padChar):
    file = open(corpusFileName,'r', encoding='utf-8') # ???????
    programs = file.read().split(corpusSplitString)
    symbols = getAlphabet(programs)
    print("symbols: ",len(symbols)," programs:",len(programs))

    assert startChar not in symbols 
    assert endChar not in symbols 
    assert unkChar not in symbols 
    assert padChar not in symbols
    charset = [startChar,endChar,unkChar,padChar] + [c for c in sorted(symbols) if symbols[c] > symbolCountThreshold]
    char2id = { c:i for i,c in enumerate(charset)}
    
    corpus = []
    for i,s in enumerate(programs):
        corpus.append( [startChar] + [ s[i] for i in range(min(len(s),maxProgramLength)) ] + [endChar] )

    testCorpus, trainCorpus  = splitSentCorpus(corpus, testFraction = 0.01)
    print('Corpus loading completed.')
    return testCorpus, trainCorpus, char2id
