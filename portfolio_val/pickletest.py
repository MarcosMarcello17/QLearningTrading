import pickle
import numpy as np
import matplotlib.pyplot as plt

archivo = pickle.load(open('50-train-Sharpe.p','rb'))
print(np.mean(archivo))
print(archivo)
sum = 0
for eps in archivo:
    if(eps >= 20000):
        sum += 1
print(sum)
print(len(archivo))
print(sum/len(archivo))
X = np.arange(1, 51, 1)
y = archivo
fig, ax = plt.subplots()
ax.plot(X, y, color='red')
plt.title('Prueba con Funcion de Ganancia')
plt.xlabel("Episodio")
plt.ylabel("Valor final")
ax.axhline(20000, 0, 50)
plt.show()
