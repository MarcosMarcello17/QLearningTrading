import pickle
import numpy as np

archivo = pickle.load(open('portfolio_val/50-train-profit.p','rb'))
#print(np.mean(archivo))
print(archivo)
