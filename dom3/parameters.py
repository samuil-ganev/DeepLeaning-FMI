import torch

corpusFileName = 'corpusFunctions'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 45 # 32
char_emb_size = 70 # 32

hid_size = 210 # 128
lstm_layers = 3 # 2
dropout = 0.10 # 0.5

epochs = 1 # 3
learning_rate = 0.007 # тренирането е на 9 епохи, като на последната lr = 0.0045

defaultTemperature = 0.42
