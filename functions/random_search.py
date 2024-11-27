from slim_gsgp.main_slim import slim
from slim_gsgp.main_gsgp import gsgp 
from slim_gsgp.main_gp import gp
from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.evaluators.fitness_functions import rmse
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from test_funcs import *

# -------------------------------- SLIM --------------------------------

def random_search_slim(X,y,scale=False, p_train=0.7, p_val=0.5,
                       iterations=50, pop_size=100, n_iter=100,
                       struct_mutation=False):
    
    """"
    Perform a random search for the best hyperparameters for the SLIM algorithm.

    Arguments
    ---------
    X: torch.tensor
        The input data.
    y: torch.tensor 
        The target data.
    scale: bool
        Whether to scale the data or not.
    p_train: float
        The percentage of the training set.
    p_val: float
        The percentage of the validation set (from the test set).
    iterations: int
        The number of iterations to perform.
    pop_size: int
        The population size.
    n_iter: int
        The number of iterations to perform.
    struct_mutation: bool
        Whether to use structural mutation or not.

    Returns
    -------
    results_slim: dict
        A dictionary containing the best hyperparameters for each SLIM algorithm.
    """
    params = {
    'p_inflate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'max_depth': [12,13,14,15,16,17,18,19,20,21,22,23,24],
    'init_depth': [5,6,7,8,9,10,11,12],
    'prob_const': [0.05, 0.1, 0.15, 0.2],
    'p_prune': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'tournament_size': [2, 3, 4],
    'p_xo': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] if struct_mutation==True else [0,0],
    'p_struct_xo': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if struct_mutation==True else [0,0],
    'prob_replace': [0.01, 0.015, 0.02, 0.025] if struct_mutation==True else [0,0],
    }

    # Perform a split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=1-p_val)

    if scale:
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
        X_val = torch.tensor(scaler_x.transform(X_val), dtype=torch.float32)
        X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
        y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
        y_val = torch.tensor(scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
        y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
        
    print("Training set size: ", X_train.shape[0])
    print("Validation set size: ", X_val.shape[0])
    print("Test set size: ", X_test.shape[0])

    results_slim = {}
    for algorithm in ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]:
        results = {}
        for i in tqdm(range(iterations)):
            p_inflate = np.random.choice(params['p_inflate'])
            max_depth = int(np.random.choice(params['max_depth']))
            init_depth = int(np.random.choice(params['init_depth']))
            tournament_size = int(np.random.choice(params['tournament_size']))  
            prob_const = np.random.choice(params['prob_const'])
            prob_replace = np.random.choice(params['prob_replace'])
            p_prune = np.random.choice(params['p_prune'])
            p_xo = np.random.choice(params['p_xo'])
            p_struct_xo = np.random.choice(params['p_struct_xo'])

            if init_depth + 6 > max_depth:
                max_depth = init_depth + 6

            slim_ = slim(X_train=X_train, y_train=y_train, dataset_name='dataset_1',
                            X_test=X_val, y_test=y_val, slim_version=algorithm, pop_size=pop_size, n_iter=n_iter,
                            ms_lower=0, ms_upper=1, p_inflate=p_inflate, max_depth=max_depth, init_depth=init_depth, 
                            seed=20, prob_const=prob_const, n_elites=1, log_level=0, verbose=0,
                            struct_mutation=struct_mutation, prob_replace=prob_replace, p_prune=p_prune, 
                            p_xo=p_xo, p_struct_xo=p_struct_xo, tournament_size=tournament_size,
                            )

            predictions_slim = slim_.predict(X_test)
            rmse_score = float(rmse(y_true=y_test, y_pred=predictions_slim))
            results[rmse_score] = {
                'p_inflate': p_inflate,
                'max_depth': max_depth,
                'init_depth': init_depth,
                'prob_const': prob_const,
                'prob_replace': prob_replace,
                'p_prune': p_prune,
                'p_xo': p_xo,
                'p_struct_xo': p_struct_xo,
            }

        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}
        # Get the best hyperparameters
        best_hyperparameters = list(results.values())[0]
        results_slim[algorithm] = best_hyperparameters
    return results_slim



# -------------------------------- GSGP --------------------------------

def random_search_gsgp(X, y, scale=False, p_train=0.7, p_val=0.5, iterations=50, 
                       pop_size=100, n_iter=100, verbose=0, threshold=100000):
    """
    Perform a random search for the best hyperparameters for the GSGP algorithm.

    Arguments
    ---------
    X: torch.tensor
        The input data.
    y: torch.tensor     
        The target data.
    scale: bool
        Whether to scale the data or not.
    p_train: float
        The percentage of the training set.
    p_val: float
        The percentage of the validation set (from the test set).
    iterations: int
        The number of iterations to perform.
    pop_size: int
        The population size.
    n_iter: int
        The number of iterations to perform.
    verbose: int
        The verbosity level.
    threshold: int
        The maximum number of nodes allowed in the tree.

    Returns
    -------
    best_hyperparameters: dict
        A dictionary containing the best hyperparameters for the GSGP algorithm
    """
    
    # Define parameter space
    params = {
        'p_xo': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'init_depth': [5, 6, 7, 8, 9, 10, 11, 12],
        'prob_const': [0.05, 0.1, 0.15, 0.2],
        'tournament_size': [2, 3, 4, 5],
    }

    # Dataset split
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=1-p_val)

    if scale:
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
        X_val = torch.tensor(scaler_x.transform(X_val), dtype=torch.float32)
        X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
        y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
        y_val = torch.tensor(scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
        y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

    print("Training set size:", X_train.shape[0])
    print("Validation set size:", X_val.shape[0])
    print("Test set size:", X_test.shape[0])

    # Random search loop
    results = {}
    for i in tqdm(range(iterations)):
        # Randomly select parameters
        p_xo = np.random.choice(params['p_xo'])
        init_depth = int(np.random.choice(params['init_depth']))
        prob_const = np.random.choice(params['prob_const'])
        tournament_size = int(np.random.choice(params['tournament_size']))

        # Run GSGP
        try:
            gsgp_model = gsgp(
                X_train=X_train, y_train=y_train,
                X_test=X_val, y_test=y_val,
                pop_size=pop_size, n_iter=n_iter,
                p_xo=p_xo, init_depth=init_depth,
                prob_const=prob_const, tournament_size=tournament_size,
                dataset_name='random_search_gsgp',
                verbose=verbose, log_level=0, minimization=True, reconstruct=True,
            )

            if gsgp_model.nodes > threshold:
                # Skip if the tree is too large
                print(f"Tree too large: {gsgp_model.nodes}")
                continue

            # Predict and evaluate
            predictions = gsgp_model.predict(X_test)
            rmse_score = float(rmse(y_true=y_test, y_pred=predictions))
            print(rmse_score)

            # Store results
            results[rmse_score] = {
                'p_xo': p_xo,
                'init_depth': init_depth,
                'pop_size': pop_size,
                'n_iter': n_iter,
                'prob_const': prob_const,
                'tournament_size': tournament_size,
            }
        except Exception as e:
            print(f"Iteration {i} failed with error: {e}")
            continue

    # Sort results by RMSE and return the best configuration
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}
    best_hyperparameters = list(results.values())[0] if results else {}
    return best_hyperparameters