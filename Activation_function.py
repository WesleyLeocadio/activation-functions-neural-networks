import numpy as np

def stepFunction(soma):
    if(soma>=1):
        return 1
    return 0
test = stepFunction(-5)

#retorna valores entre 0 e 1. Essa função é muito utilizada para retornar probabilidades. Pode ser 
# como resultado final de uma rede neural quando se quer fazer a classificação de problemas binários
def sigmoid(soma):
    return 1/(1+np.exp(-soma))

test = sigmoid(0.358)

# Função tangente hiperbólica retorna valores entre -1 e 1 também pode ser utilizada para classificação
#quando se tem apenas duas classes e uma das vantagens é que as entradas negativas serão mapeadas fortementes 
#negativas
def tahnFunction(soma):
    return (np.exp(soma)- np.exp(-soma))/(np.exp(soma)+ np.exp(-soma))

test = tahnFunction(-0.358)


def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0
test = reluFunction(0.358)


#retorna o valor passado, ou seja, essa função nao faz nada. É muito utilizada em problemas de regressao. Ex: quando
#quando quer prever o valor de um carro, se o preço é dez mil não posso aplicar uma função 
def linearFunction(soma):
    return soma

test = linearFunction(-0.358)


#Muito utilizada em deep learning quando se tem problemas com mais de duas classes, retorna as probabilidades
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

valores = [7.0, 2.0, 1.3]
print(softmaxFunction(valores))

