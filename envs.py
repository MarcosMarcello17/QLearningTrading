import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools
import statistics


class TradingEnv(gym.Env):
  def __init__(self, train_data, risk_free_data, init_invest, utility_function):
    self.stock_price_history = np.around(train_data)
    self.stock_rfree_history = np.around(risk_free_data,1)
    self.n_stock, self.n_step = self.stock_price_history.shape
    self.utility = utility_function

    self.init_invest = init_invest
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.stock_free_price = None
    self.cash_in_hand = None
    self.val_act = None
    self.transaction_cost = 0.005
    self.return_tot = None
    self.historic_return = None
    
    self.action_space = spaces.Discrete(3**self.n_stock)
    
    stock_max_price = self.stock_price_history.max(axis=1)
    stock_free_max_price = np.max(self.stock_rfree_history)
    stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
    stock_free_price_range = [[0, stock_free_max_price]]
    price_range = [[0, mx] for mx in stock_max_price]
    cash_in_hand_range = [[0, init_invest * 2]]
    self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range + stock_free_price_range)

    self._reset()


  def _reset(self):
    self.cur_step = 0
    self.stock_owned = 0
    self.stock_price = self.stock_price_history[:, self.cur_step]
    self.stock_free_price = self.stock_rfree_history[self.cur_step]
    self.cash_in_hand = self.init_invest
    self.return_tot = 0
    self.historic_return = []
    self._get_val()
    return self._get_obs()
  
  def _calcVariables(self, action):
    prev_price = self.stock_price #z_t-1
    F_t_prev = 0 #F_t-1
    if (self.stock_owned > 0):
      F_t_prev = 1
    self.cur_step += 1
    self.stock_price = self.stock_price_history[:, self.cur_step]
    self.stock_free_price = self.stock_rfree_history[self.cur_step]
    risk_free_variance = self.stock_rfree_history[self.cur_step] - self.stock_rfree_history[self.cur_step - 1] #r^f_t
    self._trade(action)
    F_t_cur = 0 #F_t
    if (self.stock_owned > 0):
      F_t_cur = 1
    cur_price = self.stock_price #z_t
    r_t = cur_price - prev_price #r_t
    owning_change = abs(F_t_cur - F_t_prev)
    return r_t, risk_free_variance, F_t_prev, owning_change


  def difSharpeRatio(self):
    n = len(self.historic_return)
    A_t = (1/(n-1)) * np.sum(self.historic_return[:-1])
    sum_square = []
    for i in range(0, n):
      sum_square.append(pow(self.historic_return[i],2))
    B_t = ((1/(n-1)) * np.sum(sum_square[:-1]))
    delta_A = self.historic_return[-1] - A_t
    delta_B = pow(self.historic_return[-1], 2) - B_t
    D_t = ((B_t * delta_A) - (0.5*A_t*delta_B))/pow(B_t - pow(A_t, 2), 3/2)
    return D_t

  def _step(self, action):
    assert self.action_space.contains(action)
    r_t, r_ft, f_t_prev, ownChange = self._calcVariables(action)
    self._get_val()
    cur_val = self.val_act
    return_t = self.stock_owned *(r_ft + f_t_prev*(r_t - r_ft) - (self.transaction_cost * self.stock_price)*ownChange)
    self.historic_return.append(return_t[0])
    if self.utility == 'Profit':
      reward = return_t[0]
      self.return_tot = self.return_tot + reward
    if self.utility == 'Sharpe':
      if len(self.historic_return) > 1:
        if statistics.stdev(self.historic_return) == 0:
          sr_now = statistics.mean(self.historic_return) / 1
          reward = sr_now
        else:
          sr_now = statistics.mean(self.historic_return) / statistics.stdev(self.historic_return)
          reward = self.difSharpeRatio()
        self.return_tot = sr_now
      else:
        reward = 0
        self.return_tot = 0
      
    done = self.cur_step == self.n_step - 1
    info = {'cur_val': cur_val}
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = []
    obs.append(self.stock_owned)
    obs.extend(list(self.stock_price))
    obs.append(self.cash_in_hand)
    obs.append(self.stock_free_price)
    return obs


  def _get_val(self):
    self.val_act = np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand



  def _trade(self, action):
    action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
    action_vec = action_combo[action]

    for i, a in enumerate(action_vec):
      if a == 0:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned
        self.stock_owned = 0
      elif a == 2:
        can_buy = True
        while can_buy:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned += 1
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False