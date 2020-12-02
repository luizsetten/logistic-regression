from fileHandler import readFile, separator
from Regressao_Logistica import constroi_modelo
import numpy as np

matrix = readFile('tic-tac-toe.data')

X_treino, Y_treino, X_teste, Y_teste = separator(0.3, matrix)

X_treino = np.array(X_treino).T.astype(np.float)
Y_treino = np.array([Y_treino]).astype(np.float)
X_teste = np.array(X_teste).T.astype(np.float)
Y_teste = np.array([Y_teste]).astype(np.float)

print(X_treino.shape)
print(Y_treino.shape)
print(X_teste.shape)
print(Y_teste.shape)



constroi_modelo(X_treino, Y_treino, X_teste, Y_teste, 50, 0.1, print_custo=False)