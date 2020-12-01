from fileHandler import readFile, separator
from Regressao_Logistica import constroi_modelo
import numpy as np

matrix = readFile('tic-tac-toe.data')

X_treino, Y_treino, X_teste, Y_teste = separator(0.3, matrix)

print(np.array(X_treino).shape[0])

constroi_modelo(np.array(X_treino).astype(np.float), np.array(Y_treino).astype(np.float), np.array(X_teste).astype(np.float), np.array(Y_teste).astype(np.float), 50, 0.1, print_custo=False)