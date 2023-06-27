import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/allanspadini/numpy/dados/citrus.csv'
dado = np.loadtxt(url, delimiter=',',usecols=np.arange(1,6,1),skiprows=1)

diametro_laranja = dado[:5000, 0]
peso_laranja = dado[:5000, 1]

diametro_toranja = dado[5000:, 0]
peso_toranja = dado[5000:, 1]

plt.plot(diametro_laranja, peso_laranja)
plt.plot(diametro_toranja, peso_toranja)

Y = peso_laranja
X = diametro_laranja
n = np.size(diametro_laranja)

a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y))/(n*np.sum(X**2) - np.sum(X)**2)
b = np.mean(Y) - a*np.mean(X)
x = diametro_laranja
y = a*x+b

plt.plot(x, y)

Z = peso_toranja
W = diametro_toranja
m = np.size(diametro_toranja)

c = (n*np.sum(W*Z) - np.sum(W)*np.sum(Z))/(n*np.sum(W**2) - np.sum(W)**2)
d = np.mean(Z) - a*np.mean(W)
i = diametro_toranja
j = c*i+d

plt.plot(i, j)

plt.show()

norma = np.array([])
np.random.seed(84)
coef_angulares = np.random.uniform(low=0.0,high=30.0,size=100)
b=17
for i in range(100):
  norma = np.append(norma,np.linalg.norm(Y- (coef_angulares[i]*X+b)))
