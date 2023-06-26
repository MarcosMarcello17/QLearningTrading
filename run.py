import pickle
import time
import numpy as np
import argparse
import re

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir, get_data_f



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode', type=int, default=2000,
                      help='Cantidad de episodios')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='Batch size')
  parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                      help='Inversion inicial')
  parser.add_argument('-u', '--utility', type=str, default='Profit', help="Funcion a utilizar")
  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')

  timestamp = time.strftime('%Y%m%d%H%M')

  data = np.around(get_data('data/daily_IBM.csv'))
  data_f = get_data_f()
  limit = 141
  train_data = data[:, :limit]
  test_data = data[:, limit:]
  train_data_f = data_f[:limit]

  env = TradingEnv(train_data, train_data_f, args.initial_invest, utility_function=args.utility)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  portfolio_value = []

  for e in range(args.episode):
    state = env.reset()
    state = scaler.transform([state])
    for time in range(env.n_step):
      print(time)
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      next_state = scaler.transform([next_state])
      agent.remember(state, action, reward, next_state, done)
      state = next_state
      if done:
        print("episode: {}/{}, episode end value: {}".format(
          e + 1, args.episode, info['cur_val']))
        portfolio_value.append(info['cur_val'])
        break
      if len(agent.memory) > args.batch_size:
        agent.replay(args.batch_size)

  with open('portfolio_val/{}-{}-{}.p'.format(args.episode, 'train', args.utility), 'wb') as fp:
    pickle.dump(portfolio_value, fp)