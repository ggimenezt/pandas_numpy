import numpy as np
import matplotlib.pyplot as plt

#carregar arquivo com informações
dado = np.loadtxt('Numpy/numpy-dados/apples_ts.csv', delimiter=',', usecols=np.arange(1, 88, 1)) 

#número de dimensões
print(dado.ndim)
#número de dados
print(dado.size)
#dimensões do array
print(dado.shape)

#Realizar transposição
dado_transposto = dado.T

#Separa os dados por coluna
#datas = dado_transposto[:,0]
datas = np.arange(1,88,1)
precos = dado_transposto[:,1:6]

#Separando por cidades
Moscow = precos[:,0]
Kaliningrad = precos[:,1]
Petersburg = precos[:,2]
Krasnodar = precos[:,3]
EKaterinburg = precos[:,4]

#Plota o grafico
#plt.plot(datas, precos[:,0])
#plt.show()

#Separando por anos
Moscow_ano1 = Moscow[0:12]
Moscow_ano2 = Moscow[13:25]
Moscow_ano3 = Moscow[25:37]
Moscow_ano4 = Moscow[37:49]

#Plotando grafico com anos
#plt.plot(np.arange(1, 13, 1), Moscow_ano1)
#plt.plot(np.arange(1, 13, 1), Moscow_ano2)
#plt.plot(np.arange(1, 13, 1), Moscow_ano3)
#plt.plot(np.arange(1, 13, 1), Moscow_ano4)
#plt.legend(['ano1', 'ano2', 'ano3', 'ano4'])
#plt.show()

#Comparando igualdade de arrays
np.array_equal(Moscow_ano3, Moscow_ano4) # = False

#Conferindo se existe uma diferença maior que X entre os arrays
np.allclose(Moscow_ano3, Moscow_ano4, 10) # = True

#NaN = not a number
#Verificando a quantidade de NaNs

#Lidando com os nans

#1.Interpolar
Kaliningrad[4] = np.mean([Kaliningrad[3], Kaliningrad[5]])

#Comparando médias
media_Moscow = np.mean(Moscow)
media_Kaliningrad = np.mean(Kaliningrad)

if (media_Kaliningrad <= media_Moscow):
    print("Preço médio de Moscow é mais alto")

else:
    print("Preço médio de Kaliningrad é mais alto")

#Analisando dados
plt.plot(datas, precos[:,0])

#Ajustar uma reta para os dados
a = 2
b = 80
x = datas
y = a*x+b

#Calculando a diferença do nosso array com o array de comparação para ajuste e resumindo ele em um número
dif = np.sqrt(np.sum(np.power(Moscow - y, 2))) #Muito alto, chuta um número mais baixo

a = 0.52
b = 80
x = datas
y = a*x+b

dif = np.sqrt(np.sum(np.power(Moscow - y, 2))) #Muito melhor

#Todo calculo resumido (caluclo da norma entre 2 arrays)
np.linalg.norm(Moscow-y)

#plt.plot(x, y)
#plt.show()

#Ao invés de chutar os coeficientes, nós podemos calcular esses valores
#â=(n*Soma(x*Y))-Soma(X)*Soma(Y)/n(Soma(X²))-(Soma(X))²
#a=coeficiente angular, n=número de elementos, Y=Moscow, X=Datas

n = np.size(Moscow)
Y = Moscow
X = datas

a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y))/(n*np.sum(X**2) - np.sum(X)**2)
print(a)

#b=Media(Y)-â*Media(X)
#b=coeficiente linear

b = np.mean(Y) - a*np.mean(X)
print(b)

y = a*x+b

plt.plot(x, y)
#coloca um ponto na reta através da equação que encontramos
plt.plot(41.5, 41.5*a+b, "*r")
plt.show()

dif = np.linalg.norm(Moscow-y)

print(dif)

#Gerção de números aleatórios para gerar os coeficientes

#função que gera um intervalo de inteiros aleatórios

np.random.randint(low=40, high=100, size=100)

coef_ang = np.random.uniform(low=0.10, high=0.90, size=100)

norma2 = np.array([])
for i in range(100):
    norma2 = np.append(norma2, np.linalg.norm(Moscow-(coef_ang[i]*X+b)))

#encontra o menor valor nessearray, é o mais próximo do coef angular

#precisamos de reprodutibilidade nos números aleatórios
np.random.seed(84)
coef_ang = np.random.uniform(low=0.10, high=0.90, size=100)
norma2 = np.array([])
for i in range(100):
    norma2 = np.append(norma2, np.linalg.norm(Moscow-(coef_ang[i]*X+b)))
#salvando nossos resultados
dados = np.column_stack([norma2, coef_ang]) #agrega arrays unidimensionais em arrays bidimensionais
np.savetxt('dados.csv', dados, delimiter=',')