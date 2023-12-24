#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################

import langmodel
import nltk
import a1
from nltk.corpus import PlaintextCorpusReader

#############################################################################
#### Начало на тестовете
#### ВНИМАНИЕ! Тези тестове са повърхностни и тяхното успешно преминаване е само предпоставка за приемането, но не означава задължително, че програмата Ви ще бъде приета. За приемане на заданието Вашата програма ще бъде подложена на по-задълбочена серия тестове.
#############################################################################

L1 = ['заявката','заявката','заявката','заявката','заявката','заявката']
L2 = ['заявката','язвката','заявьата','завякатва','заявкатаа','вя']
C = [0,3,1,2,1,7]
D = [0.0,8.0,2.5,5.25,3.0,20.25]

#### Тест на editDistance
for s1,s2,d in zip(L1,L2,C):
    assert a1.editDistance(s1,s2)[-1,-1] == d, "Разстоянието между '{}' и '{}' следва да е '{}'".format(s1,s2,d)
print("Функцията editDistance премина теста.")

#### Тест на editWeight
dummy_weights = {}
for a in langmodel.alphabet:
	dummy_weights[(a,a)] = 0.0
	dummy_weights[(a,'')] = 3.0
	dummy_weights[('',a)] = 3.0
	for b in langmodel.alphabet:
		if a != b:
			dummy_weights[(a,b)] = 2.5
		dummy_weights[(a+b,b+a)] = 2.25

for s1,s2,d in zip(L1,L2,D):
    assert a1.editWeight(s1,s2,dummy_weights) == d, "Теглото между '{}' и '{}' следва да е '{}'".format(s1,s2,d)
print("Функцията editWeight премина теста.")

#### Тест на generate_edits
assert len(set(a1.generateEdits("тест"))-set(["тест"])) == 287, "Броят на елементарните редакции \"тест\"  следва да е 287"
print("Функцията generateEdits премина теста.")


#### Тест на bestAlignment
def test_bestAlignment(s1,s2,alignment):
	a1 = ''
	a2 = ''
	w = 0
	for u,v in alignment:
		if len(u)==1 and len(v)==1:
			if u!=v: w+=1
		elif len(u)==0 and len(v)==1: w+=1
		elif len(u)==1 and len(v)==0: w+=1
		elif len(u)==2 and len(v)==2 and u[0]==v[1] and u[1]==v[0]: w+=1
		else:
			w=None
			break
			
		a1 += u
		a2 += v
	if a1 != s1 or a2 != s2: w=None
	return w

for s1,s2,d in zip(L1,L2,C):
    assert test_bestAlignment(s1,s2,a1.bestAlignment(s1,s2)) == d, "Грешно минимално подравняване между '{}' и '{}'".format(s1,s2)
print("Функцията bestAlignment премина теста.")


#### Тест на correct_spelling

print('Прочитане на корпуса от текстове...')
corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
fullSentCorpus = [ [langmodel.startToken] + [w.lower() for w in sent] + [langmodel.endToken] for sent in myCorpus.sents()]
print('Готово.')

print('Трениране на Марковски езиков модел...')
M2 = langmodel.MarkovModel(fullSentCorpus,2)
print('Готово.')

print('Прочитане на корпуса със правописни грешки...')
with open('corpus') as f: 
	lines = f.read().split('\n')[:-1]
error_corpus = [c.split('\t') for c in lines]
print('Готово.')

weights = a1.trainWeights(error_corpus)

assert a1.correctSpelling("светфно по футбол",M2,weights,1.0) == 'световно по футбол', "Коригираната заявка следва да е 'световно по футбол'"
print("Функцията correctSpelling премина теста.")

