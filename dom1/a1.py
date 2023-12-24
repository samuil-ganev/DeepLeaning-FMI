#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################

### Домашно задание 1
###
### За да работи програмата трябва да се свали корпус от публицистични текстове за Югоизточна Европа,
### предоставен за некомерсиално ползване от Института за български език - БАН
###
### Корпусът може да бъде свален от:
### Отидете на http://dcl.bas.bg/BulNC-registration/#feeds/page/2
### И Изберете:
###
### Корпус с новини
### Корпус от публицистични текстове за Югоизточна Европа.
### 27.07.2012 Български
###	35337  7.9M
###
### Архивът трябва да се разархивира в директорията, в която е програмата.
###
### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii
###
### Ако все още нямате създадено обкръжение прочетете файла README.txt за инструкции

import langmodel
import math
import numpy as np



def editDistance(s1, s2):
    #### функцията намира разстоянието на Левенщайн - Дамерау между два низа
    #### вход: низовете s1 и s2
    #### изход: матрицата M с разстоянията между префиксите на s1 и s2 (виж по-долу)

    M = np.zeros((len(s1) + 1, len(s2) + 1))
    #### M[i,j] следва да съдържа разстоянието между префиксите s1[:i] и s2[:j]
    #### M[len(s1),len(s2)] следва да съдържа разстоянието между низовете s1 и s2
    #### За справка разгледайте алгоритъма editDistance от слайдовете на Лекция 1

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

    M[1:, 0] = np.array([i for i in range(1, len(s1) + 1)])
    M[0, 1:] = np.array([i for i in range(1, len(s2) + 1)])

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            paths = []
            paths.append(M[i - 1, j] + 1)
            paths.append(M[i, j - 1] + 1)
            paths.append(M[i - 1, j - 1] + (1 if s1[i - 1] != s2[j - 1] else 0))
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                paths.append(M[i - 2, j - 2] + 1)
            M[i, j] = min(paths)

    #### Край на Вашия код
    #############################################################################

    return M


def editWeight(s1, s2, Weight):
    #### функцията editWeight намира теглото между два низа
    #### вход: низовете s1 и s2, както и речник Weight, съдържащ теглото на всяка от елементарните редакции
    #### изход: минималната сума от теглата на елементарните редакции, необходими да се получи от единия низ другия

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

    W = np.zeros((len(s1) + 1, len(s2) + 1))

    W[1:, 0] = np.array([i * Weight[(s1[i - 1], '')] for i in range(1, len(s1) + 1)])
    W[0, 1:] = np.array([i * Weight[('', s2[i - 1])] for i in range(1, len(s2) + 1)])

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            paths = []
            paths.append(W[i - 1, j] + Weight[(s1[i - 1], '')])
            paths.append(W[i, j - 1] + Weight[('', s2[j - 1])])
            paths.append(W[i - 1, j - 1] + Weight[(s1[i - 1], s2[j - 1])])
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                paths.append(W[i - 2, j - 2] + Weight[(f'{s1[i - 1]}{s1[i - 2]}', f'{s2[j - 1]}{s2[j - 2]}')])
            W[i, j] = min(paths)

    return W[-1, -1]


#### Край на Вашия код
#############################################################################


def bestAlignment(s1, s2):
    #### функцията намира подравняване с минимално тегло между два низа
    #### вход:
    ####     низовете s1 и s2
    #### изход:
    ####     списък от елементарни редакции, подравняващи s1 и s2 с минимално тегло

    M = editDistance(s1, s2)
    alignment = []

    #############################################################################
    #### УПЪТВАНЕ:
    #### За да намерите подравняване с минимално тегло следва да намерите път в матрицата M,
    #### започващ от последния елемент на матрицата -- M[len(s1),len(s2)] до елемента M[0,0].
    #### Всеки преход следва да съответства на елементарна редакция, която ни дава минимално
    #### тегло, съответстващо на избора за получаването на M[i,j] във функцията editDistance.
    #### Събирайки съответните елементарни редакции по пъта от M[len(s1),len(s2)] до M[0,0]
    #### в обратен ред ще получим подравняване с минимално тегло между двата низа.
    #### Всяка елементарна редакция следва да се представи като двойка низове.
    #### ПРЕМЕР:
    #### bestAlignment('редакция','рдацкиа') = [('р','р'),('е',''),('д' 'д'),('а','а'),('кц','цк'),('и','и'),('я','а')]
    #### ВНИМАНИЕ:
    #### За някой двойки от думи може да съществува повече от едно подравняване с минимално тегло.
    #### Достатъчно е да изведете едно от подравняванията с минимално тегло.
    #############################################################################

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

    i, j = len(s1), len(s2)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            alignment.append((s1[i - 1], s1[i - 1]))
            i -= 1
            j -= 1
        elif i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1] and M[i, j] == M[i - 2, j - 2] + 1:
            alignment.append((s1[i - 2:i], s1[i - 2:i][::-1]))
            i -= 2
            j -= 2
        elif i > 0 and M[i, j] == M[i - 1, j] + 1:
            alignment.append((s1[i - 1], ''))
            i -= 1
        elif j > 0 and M[i, j] == M[i, j - 1] + 1:
            alignment.append(('', s2[j - 1]))
            j -= 1
        else:
            alignment.append((s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1

    alignment = alignment[::-1]

    #### Край на Вашия код
    #############################################################################

    return alignment


def trainWeights(corpus):
    #### Функцията editionWeights връща речник съдържащ теглото на всяка от елементарните редакции
    #### Функцията реализира статистика за честотата на елементарните редакции от корпус, състоящ се от двойки сгрешен низ и коригиран низ. Теглата са получени след оценка на вероятността за съответната грешка, използвайки принципа за максимално правдоподобие.
    #### Вход: Корпус от двойки сгрешен низ и коригиран низ
    #### изход: речник съдържащ теглото на всяка от елементарните редакции

    opCount = {}

    ids = subs = ins = dels = trs = 0
    for q, r in corpus:
        alignment = bestAlignment(q, r)
        for op in alignment:
            if len(op[0]) == 1 and len(op[1]) == 1 and op[0] == op[1]:
                ids += 1
            elif len(op[0]) == 1 and len(op[1]) == 1:
                subs += 1
            elif len(op[0]) == 0 and len(op[1]) == 1:
                ins += 1
            elif len(op[0]) == 1 and len(op[1]) == 0:
                dels += 1
            elif len(op[0]) == 2 and len(op[1]) == 2:
                trs += 1
    N = ids + subs + ins + dels + trs

    weight = {}
    for a in langmodel.alphabet:
        weight[(a, a)] = - math.log(ids / N)
        weight[(a, '')] = - math.log(dels / N)
        weight[('', a)] = - math.log(ins / N)
        for b in langmodel.alphabet:
            if a != b:
                weight[(a, b)] = - math.log(subs / N)
            weight[(a + b, b + a)] = - math.log(trs / N)
    return weight


def generateEdits(q):
    ### помощната функция, generate_edits по зададена заявка генерира всички възможни редакции на разстояние едно от тази заявка.
    ### Вход: заявка като низ q
    ### Изход: Списък от низове на Левенщайн - Дамерау разстояние 1 от q
    ###
    ### В тази функция вероятно ще трябва да използвате азбука, която е дефинирана в langmodel.alphabet
    ###
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-15 реда

    ins_chr_strings = [q[:i] + ch + q[i:] for i in range(len(q) + 1) for ch in langmodel.alphabet]
    del_chr_strings = [q[:i] + q[i + 1:] for i in range(len(q))]
    subs_chr_string = [q[:i] + ch + q[i + 1:] for i in range(len(q)) for ch in langmodel.alphabet]
    tr_chrs_strings = [q[:i] + q[i + 1] + q[i] + q[i + 2:] for i in range(len(q) - 1)]
    return ins_chr_strings + del_chr_strings + subs_chr_string + tr_chrs_strings



#### Край на Вашия код
#############################################################################


def generateCandidates(query, dictionary):
    ### Започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, за да се получат кандидатите за корекция. Връщат се единствено кандидати, за които всички думи са в речника dictionary.

    ### Вход:
    ###	 Входен низ: query
    ###	 Речник: dictionary

    ### Изход:
    ###	 Списък от низовете, които са кандидати за корекция

    def allWordsInDictionary(q):
        ### Помощна функция, която връща истина, ако всички думи в заявката са в речника
        return all(w in dictionary for w in q.split())

    L = []
    if allWordsInDictionary(query):
        L.append(query)
    for query1 in generateEdits(query):
        if allWordsInDictionary(query1):
            L.append(query1)
        for query2 in generateEdits(query1):
            if allWordsInDictionary(query2):
                L.append(query2)
    return L


def correctSpelling(r, model, weights, mu=1.0, alpha=0.9):
    ### Комбинира вероятността от езиковия модел с вероятността за редактиране на кандидатите за корекция, генерирани от generate_candidates за намиране на най-вероятната желана (коригирана) заявка по дадената оригинална заявка query.
    ###
    ### Вход:
    ###	    заявка: r,
    ###	    езиков модел: model,
    ###     речник съдържащ теглото на всяка от елементарните редакции: weights
    ###	    тегло на езиковия модел: mu
    ###	    коефициент за интерполация на езиковият модел: alpha
    ### Изход: най-вероятната заявка

    ### УПЪТВАНЕ:
    ###    Удачно е да работите с логаритъм от вероятностите. Логаритъм от вероятността от езиковия модел може да получите като извикате метода model.sentenceLogProbability. Минус логаритъм от вероятността за редактиране може да получите като извикате функцията editWeight.
    #############################################################################
    #### Начало на Вашия код за основното тяло на функцията correct_spelling. На мястото на pass се очакват 3-10 реда

    dictionary = model.kgrams[tuple()]

    all_possible_q = set([q.strip() for q in generateCandidates(r.lower(), dictionary)])
    probs = []

    for q in all_possible_q:

        prob = -editWeight(r.lower(), q, weights)
        prob += mu * model.sentenceLogProbability(q.split(), alpha)

        probs.append((q, prob))
    probs.sort(key=lambda x: x[1], reverse=True)
    #print(probs)
    return probs[0][0]

#### Край на Вашия код
#############################################################################
