'''
Exemplo de:
-> como treinar o modelo Kriging;
-> como utilizar o modelo Kriging para fazer predições de varios pontos simultaneamente;
-> como utilizar o modelo Kriging para fazer predições de um ponto só;
-> como utilizar o modelo Kriging em uma otimização.
'''

'Importando bibliotecas próprias'
from Kriging.regpoly import regpoly0, regpoly1, regpoly2
from Kriging.corr import gauss
from Kriging.dacefit import dacefit
from Kriging.predictor import predictor

'Importando bibliotecas externas'
import matplotlib.pyplot as plt 
import numpy as np

'Dados'
n_x = 2 #Número de variaveis independentes
LB = np.array([-5, -5]) #Limite superior das variáveis independentes
UB = np.array([5, 5]) #Limite inferior das variáveis independentes
n_train = 200 #Número de pontos para treino do modelo 
n_test = 500 #Número de pontos para avaliação do modelo

'Função que representa o modelo rigoroso'
def process(X):
    y = 2*X[:,0]**2 + 5*X[:,1]**2 + 9*np.exp(X[:,0]) + 7*np.exp(X[:,1])
    y = np.reshape(y,[len(y),1])
    return y

'Função para gerar os pontos de amostragem'
def lhs(n_x, LB, UB, n):
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(n_x)
    sample = sampler.random(n)
    sample_scaled = qmc.scale(sample=sample, l_bounds=LB, u_bounds=UB)
    return sample_scaled    

'Gráfico do modelo rigoroso'
x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)
X1, X2 = np.meshgrid(x1, x2)
Y = 2*X1**2 + 5*X2**2 + 9*np.exp(X1) + 7*np.exp(X2)
fig, ax = plt.subplots()
CS = ax.contour(X1, X2, Y, 200)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Curvas de nivel do modelo rigoroso')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Y')

'Dados para treino'
X_train = lhs(n_x, LB, UB, n_train)
y_train = process(X_train)

'Gerando os dados de teste'
X_test = lhs(n_x, LB, UB, n_test)
y_test = process(X_test)

'Treinando o Kriging'
t0 = np.ones(n_x)*10 #Valores inicias dos parâmetros
lob = list(np.ones(n_x)*0.1) #Limites inferiores dos parâmetros
upb = list(np.ones(n_x)*20)  #Limites superiores dos parâmetros
dmodel, perf = dacefit(X_train, y_train, t0, regpoly2, gauss, lob, upb)

'Usando o Kriging para a predição dos valores de y nos pontos de teste'
y_pred_multi, dy_multi, mse_multi, dmse_multi = predictor(X_test, dmodel)

'Fazendo o gráfico para avaliar a qualidade da predição'
xy_min = min(min(y_test[:,0]), min(y_pred_multi[:,0]))
xy_max = max(max(y_test[:,0]), max(y_pred_multi[:,0]))
p1 = np.array([xy_min, xy_max])
p2 = np.array([xy_min, xy_max])
plt.figure()
plt.plot(y_test, y_pred_multi, '.r')
plt.plot(p1, p2)
plt.title('Comparação valores preditos e reais')
plt.xlabel('Valores de Y da função rigorosa')
plt.ylabel('Valores de Y estimados pelo Kriging')

'Usando o Kriging para predizer só um ponto'
ponto = np.array([[0, 0]])
y_pred_single, dy_single, mse_single, dmse_single = predictor(ponto, dmodel)
y_real_sinlge = process(ponto)
print(f'Em {ponto} predição = {y_pred_single[0,0]} e real = {y_real_sinlge[0,0]}')
print(f'Em {ponto} dY/dX1 = ', dy_single[0,0])
print(f'Em {ponto} dY/dX2 = ', dy_single[1,0])


'Função Objetivo da otimização'
def FO(x):
    ponto = np.array([x])
    y, dy, mse, dmse = predictor(ponto, dmodel)
    return y[0,0]

'Derivada da função objetivo'
def der_FO(x):
    ponto = np.array([x])
    y, dy_mat, mse, dmse = predictor(ponto, dmodel)
    dy = (np.asarray(dy_mat)).flatten()
    return dy

'Ponto inicial e valores da FO e as suas derivadas nesse ponto'
x0 = np.array([3,3])
y = FO(x0)
dy = der_FO(x0)

'Chamando o otimizador'
from scipy.optimize import Bounds, minimize
bounds = Bounds(lb=LB, ub=UB)
method = 'SLSQP'
opt_sol = minimize(FO, x0, method=method, bounds=bounds, jac=der_FO, tol=1e-10, 
                           options={'disp': True, 'maxiter': 100, 'eps':1e-6})

print(f'Solução ótima em {opt_sol.x} com valor de FO igual a {opt_sol.fun}' )

'Colocando o ótimo achado no gráfico de curvas de nivel'
x1 = np.arange(-2, 2, 0.1)
x2 = np.arange(-2, 2, 0.1)
X1, X2 = np.meshgrid(x1, x2)
Y = 2*X1**2 + 5*X2**2 + 9*np.exp(X1) + 7*np.exp(X2)
fig, ax = plt.subplots()
CS = ax.contour(X1, X2, Y, 200)
ax.plot(opt_sol.x[0], opt_sol.x[1], '+r')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Curvas de nivel do modelo rigoroso')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Y')

