from functions.test_algorithms import test_slim
from slim_gsgp_lib.datasets.data_loader import *
import numpy as np
datasets = [globals()[i] for i in globals() if 'load' in i][2:]

args = {'p_inflate': 0.3,
  'max_depth': 21,
  'init_depth': 10,
  'prob_const': 0.1,
  'prob_replace': 0.01,
  'p_prune': 0.6,
  'p_xo': 0.2,
  'p_struct_xo': 0.7}

results = {}
for dataset in [globals()[i] for i in globals() if 'load' in i][2:]:
  X, y = dataset()
  rm, ma, nrmse, r2, mae, std_rmse, time, train, test, size = test_slim(
              X=X, y=y, args_dict=args, dataset_name='test',
              ms_lower=0, ms_upper=1, n_elites=1,
              iterations=15, scale=True, algorithm='SLIM*SIG1',
              verbose=0, p_train=0.7, show_progress=True,
          )
  rm = np.array(rm)
  rm_min = 100*rm.min()
  results[dataset.__name__] = rm_min
  print(dataset.__name__, rm_min)