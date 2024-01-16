#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import sys
import torch

import utils
import generator
import train
import model
import pickle


from parameters import *

startChar = '§'
endChar = '†'
unkChar = '±'
padChar = '№'

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    testCorpus, trainCorpus, char2id =  utils.prepareData(corpusFileName, startChar, endChar, unkChar, padChar)
    pickle.dump(testCorpus, open(testDataFileName, 'wb'))
    pickle.dump(trainCorpus, open(trainDataFileName, 'wb'))
    pickle.dump(char2id, open(char2idFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and sys.argv[1] == 'train':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    trainCorpus = pickle.load(open(trainDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))

    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    if len(sys.argv)>2: lm.load(sys.argv[2], device) # ??????

    optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
    train.trainModel(trainCorpus, lm, optimizer, epochs, batchSize)
    lm.save(modelFileName)
    # print('Model perplexity (train_set): ', train.perplexity(lm, trainCorpus, batchSize)) # dobaveno ot men
    print('Model perplexity: ',train.perplexity(lm, testCorpus, batchSize))

if len(sys.argv)>1 and sys.argv[1] == 'perplexity':
    testCorpus = pickle.load(open(testDataFileName, 'rb'))
    char2id = pickle.load(open(char2idFileName, 'rb'))
    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout).to(device)
    lm.load(modelFileName,device)
    print('Model perplexity: ',train.perplexity(lm, testCorpus, batchSize))

if len(sys.argv)>1 and sys.argv[1] == 'generate':
    if len(sys.argv)>2: seed = sys.argv[2]
    else: seed = startChar

    assert seed[0] == startChar

    if len(sys.argv)>3: temperature = float(sys.argv[3])
    else: temperature = defaultTemperature
 
    char2id = pickle.load(open(char2idFileName, 'rb'))
    lm = model.LSTMLanguageModelPack(char_emb_size, hid_size, char2id, unkChar, padChar, endChar, lstm_layers=lstm_layers, dropout=dropout)
    lm.load(modelFileName,torch.device("cpu"))
    
    print(generator.generateCode(lm, char2id, seed, temperature=temperature))


