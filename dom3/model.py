#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import torch

#################################################################
####  LSTM с пакетиране на партида
#################################################################

class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName,device))

    def __init__(self, embed_size, hidden_size, word2ind, unkToken, padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        #############################################################################
        ###  Тук следва да се имплементира инициализацията на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавки за повече слоеве на РНН, влагане за автора и dropout
        #############################################################################
        #### Начало на Вашия код.

        self.word2ind = word2ind

        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers=lstm_layers)
        self.projection = torch.nn.Linear(hidden_size, len(word2ind))

        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]

        self.dropout = torch.nn.Dropout(dropout)

        #### Край на Вашия код
        #############################################################################

    def forward(self, source):
        #############################################################################
        ###  Тук следва да се имплементира forward метода на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавка за dropout и началните скрити вектори
        #############################################################################
        #### Начало на Вашия код.

        X = self.preparePaddedBatch(source)
        E = self.embed(X[:-1])

        source_lengths = [len(s) - 1 for s in source]

        outputPacked, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted=False))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        z = self.dropout(output)
        Z = self.projection(z.flatten(0, 1))
        Y_bar = X[1:].flatten(0, 1)
        H = torch.nn.functional.cross_entropy(Z, Y_bar, ignore_index=self.padTokenIdx)
        return H
    
        #### Край на Вашия код
        #############################################################################

