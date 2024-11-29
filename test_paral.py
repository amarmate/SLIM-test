from slim_gsgp_lib.main_gsgp import gsgp 
from slim_gsgp_lib.main_gp import gp
from slim_gsgp_lib.utils.utils import train_test_split
from slim_gsgp_lib.evaluators.fitness_functions import rmse
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import time
import os
from tqdm import tqdm
from functions.test_funcs import mape, nrmse, r_squared, mae, standardized_rmse


from functions.test_algorithms import *
from functions.random_search import * 
from slim_gsgp_lib.datasets.data_loader import *
import pickle

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

X,y = datasets[2]()

X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-0.7, seed=10)

start = time.time()
final_tree = gp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=100, n_iter=100, seed=10,
                    tournament_size=2, p_xo=0.6,
                      verbose=1, log_level=0, n_elites=1, n_jobs=8)
end = time.time()

print('Time:', end-start)