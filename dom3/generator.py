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

def generateCode(model, char2id, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    result = startSentence[1:]

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.

    sequence = [char2id[char] for char in startSentence]
    id2char = {v: k for k, v in char2id.items()}

    endChar = '†'

    model eval()

    with torch.no_grad():
        for iter in range(limit):
            X = torch.tensor(sequence).unsqueeze(1)
            E = model.embed(X)
            if iter != 0:
                _, (h_n, c_n) = model.lstm(E, (h_n, c_n))
            else:
                _, (h_n, c_n) = model.lstm(E)
            h = model.dropout(h_n)
            Z = model.projection(h[-1, -1, :])
            probs = torch.softmax(Z / temperature, dim=-1).numpy()
            next_char = np.random.choice(len(char2id), p=probs.ravel())
            if next_char == char2id[endChar]:
                break
            sequence.append(next_char)
            result += id2char[next_char]

    model train()

    #### Край на Вашия код
    #############################################################################

    return result
