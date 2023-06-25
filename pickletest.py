import pickle
import numpy as np

archivo = pickle.load(open('portfolio_val/202306250438-train.p','rb'))
print(np.mean(archivo))