import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data_f(col='DTB3'):
  tbill = pd.read_csv('data/DTB3.csv', usecols=[col])
  return np.array(tbill[col].values[::])

def get_data(file, col='close'):
  stock = pd.read_csv(file, usecols=[col])
  return np.array([stock[col].values[::-1]])


def get_scaler(env):
  low = [0] * (env.n_stock * 2 + 2)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 3
  max_stock_owned = max_cash // min_price
  max_risk_free_price = np.max(env.stock_rfree_history)
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)
  high.append(max_risk_free_price)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)