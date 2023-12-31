#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################
###
### Домашно задание 2
###
#############################################################################

import numpy as np

#############################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def lossAndGradient(u_w, Vt, W, q):
    ###  Векторът u_w е влагането на целевата дума. shape(u_w) = M.
    ###  Матрицата Vt представя влаганията на контекстните думи. shape(Vt) = (n+1)xM.
    ###  Първият ред на Vt е влагането на коректната контекстна дума, а
    ###  следващите n реда са влаганията на извадката от негативни контекстни думи
	###  Матрицата W е параметър с теглата на квадратичната форма. shape(W) = MxM.
    ###  Векторът q съдържа логаритъм от n по вероятностите за контекстните думи. shape(q) = n+1.
    ###
    ###  функцията връща J -- загубата в тази точка;
    ###                  du_w -- градиентът на J спрямо u_w;
    ###                  dVt --  градиентът на J спрямо Vt.
    ###                  dW --  градиентът на J спрямо W.
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 7-15 реда

    d_c = np.zeros(Vt.shape[0])
    d_c[0] = 1

    du_w = np.dot( np.dot(W.T, Vt.T), sigmoid( np.dot(Vt, np.dot(W, u_w) ) - q ) - d_c )
    dVt = np.outer( ( sigmoid( np.dot( Vt, np.dot( W, u_w ) ) - q ) - d_c ), np.dot(W, u_w) )
    dW = np.outer( np.dot(Vt.T, sigmoid( np.dot(Vt, np.dot(W, u_w) ) - q ) - d_c), u_w )

    J = -np.log( sigmoid( np.dot( np.dot(d_c.T, Vt), np.dot(W, u_w) ) - np.dot(q.T, d_c) ) )
    for j in range(1, Vt.shape[0]):
        J -= np.log( sigmoid( q[j] - np.dot(Vt[j,:], np.dot(W, u_w)) ) )

    #### Край на Вашия код
    #############################################################################

    return J, du_w, dVt, dW


def lossAndGradientCumulative(u_w, Vt, W, q):
    ###  Изчисляване на загуба и градиент за цяла партида
    ###  Тук за всяко от наблюденията се извиква lossAndGradient
    ###  и се акумулират загубата и градиентите за S-те наблюдения
    Cdu_w = []
    CdVt = []
    CJ = 0
    CdW = 0
    S = u_w.shape[0]
    for i in range(S):
        J, du_w, dVt, dW = lossAndGradient(u_w[i],Vt[i],W,q[i])
        Cdu_w.append(du_w/S)
        CdVt.append(dVt/S)
        CdW += dW/S
        CJ += J/S
    return CJ, Cdu_w, CdVt, CdW


def lossAndGradientBatched(u_w, Vt, W, q):
    ###  Изчисляване на загуба и градиент за цяла партида.
    ###  Тук едновременно се изчислява загубата и градиентите за S наблюдения.
    ###  Матрицата u_w представя влаганията на целевите думи и shape(u_w) = SxM.
    ###  Тензорът Vt представя S матрици от влагания на контекстните думи и shape(Vt) = Sx(n+1)xM.
   	###  Матрицата W е параметър с теглата на квадратичната форма. shape(W) = MxM.
    ###  Матрицата q съдържа логаритъм от n по вероятностите за контекстните думи. shape(q) = Sx(n+1).
    ###  Във всяка от S-те матрици на Vt в първия ред е влагането на коректната контекстна дума, а
    ###  следващите n реда са влаганията на извадката от негативни контекстни думи.
    ###
    ###  Функцията връща J -- загубата за цялата партида;
    ###                  du_w -- матрица с размерност SxM с градиентите на J спрямо u_w за всяко наблюдение;
    ###                  dVt --  с размерност Sx(n+1)xM -- S градиента на J спрямо Vt.
    ###                  dW --  матрица с размерност MxM -- партидния градиент на J спрямо W.
    #############################################################
    ###  От вас се очаква вместо да акумулирате резултатите за отделните наблюдения,
    ###  да използвате тензорни операции, чрез които наведнъж да получите
    ###  резултата за цялата партида. Очаква се по този начин да получите над 2 пъти по-бързо изпълнение.
    #############################################################

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-20 реда

    J = 0

    d_c = np.zeros(Vt.shape[1])
    d_c[0] = 1
    d_c = np.tile(d_c, (Vt.shape[0], 1)).T # (n+1)xS

    Wu_wt = np.dot(W, u_w.T).T[:, :, np.newaxis] # SxMx1
    sigmoid_res = sigmoid( np.squeeze( np.matmul( Vt, np.dot(W, u_w.T).T[:, :, np.newaxis] ), axis=-1).T - q.T ) # (n+1)xS

    du_w = np.squeeze( np.matmul( np.matmul( W.T, Vt.transpose((0, 2, 1)) ), (sigmoid_res - d_c).T[:, :, np.newaxis] ), axis=-1 ) / Vt.shape[0] #SxM
    dVt = (sigmoid_res.T[:, :, np.newaxis] * Wu_wt.transpose((0, 2, 1)) - d_c.T[:, :, np.newaxis] * Wu_wt.transpose((0, 2, 1)) ) / Vt.shape[0] # Sx(n+1)xM
    dW = np.dot( np.squeeze( np.matmul( Vt.transpose((0, 2, 1)), (sigmoid_res - d_c).T[:, :, np.newaxis] ), axis=-1 ).T, u_w ) / Vt.shape[0] # MxM

    for i in range(Vt.shape[0]):
        J -= np.log( sigmoid( np.dot( np.dot( d_c[:,0].T, Vt[i] ), np.dot( W, u_w[i] ) ) - np.dot (q[i].T, d_c[:,0] ) ) )
        for j in range(1, Vt.shape[1]):
            J -= np.log( sigmoid( q[i][j] - np.dot( Vt[i][j, :], np.dot( W, u_w[i] ) ) ) )
    J /= Vt.shape[0]

    #### Край на Вашия код
    #############################################################################
    return J, du_w, dVt, dW
