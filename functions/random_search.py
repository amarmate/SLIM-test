from slim_gsgp_lib.main_slim import slim
from slim_gsgp_lib.main_gsgp import gsgp 
from slim_gsgp_lib.main_gp import gp
from slim_gsgp_lib.utils.utils import train_test_split
from slim_gsgp_lib.evaluators.fitness_functions import rmse
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os 
import pickle
import threading
import functools


# -------------------------------- SLIM --------------------------------

def random_search_slim(X,y,dataset, scale=False, p_train=0.7,
                       iterations=50, pop_size=100, n_iter=100,
                       struct_mutation=False, show_progress=True, 
                       save=True, identifier=None):
    
    """"
    Perform a random search for the best hyperparameters for the SLIM algorithm.

    Arguments
    ---------
    X: torch.tensor
        The input data.
    y: torch.tensor 
        The target data.
    dataset: str
        The name of the dataset.
    scale: bool
        Whether to scale the data or not.
    p_train: float
        The percentage of the training set.
    iterations: int
        The number of iterations to perform.
    pop_size: int
        The population size.
    n_iter: int
        The number of iterations to perform.
    struct_mutation: bool
        Whether to use structural mutation or not.
    show_progress: bool
        Whether to show the progress bar or not.
    save: bool
        Whether to save the results or not.
    identifier: str
        A unique identifier for the output file.

    Returns
    -------
    results_slim: dict
        A dictionary containing the best hyperparameters for each SLIM algorithm.
    """
    params = {
    'p_inflate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'max_depth': [12,13,14,15,16,17,18,19,20,21,22],
    'init_depth': [5,6,7,8,9,10,11,12],
    'prob_const': [0.05, 0.1, 0.15, 0.2],
    'tournament_size': [2, 3],
    'ms_lower': [0, 0, 0, 0.05, 0.1],
    'ms_upper': [1, 1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0.1],
    'p_prune': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if struct_mutation==True else [0,0],
    'p_xo': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] if struct_mutation==True else [0,0],
    'p_struct_xo': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if struct_mutation==True else [0,0],
    'prob_replace': [0.01, 0.015, 0.02, 0.025] if struct_mutation==True else [0,0],
    }

    results_slim = {}
    for algorithm in ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]:
        results = {}
        for i in tqdm(range(iterations), disable=not show_progress):
            # Perform a split of the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train, seed=i)

            if scale:
                scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
                X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
                X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
                y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
                y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

            # Randomly select parameters
            p_inflate = np.random.choice(params['p_inflate'])
            max_depth = int(np.random.choice(params['max_depth']))
            init_depth = int(np.random.choice(params['init_depth']))
            tournament_size = int(np.random.choice(params['tournament_size']))  
            prob_const = np.random.choice(params['prob_const'])
            prob_replace = np.random.choice(params['prob_replace'])
            p_prune = np.random.choice(params['p_prune'])
            p_xo = np.random.choice(params['p_xo'])
            p_struct_xo = np.random.choice(params['p_struct_xo'])
            ms_lower = int(np.random.choice(params['ms_lower']))
            ms_upper = int(np.random.choice(params['ms_upper']))

            if init_depth + 6 > max_depth:
                max_depth = init_depth + 6

            slim_ = slim(X_train=X_train, y_train=y_train, dataset_name='dataset_1',
                            X_test=X_test, y_test=y_test, slim_version=algorithm, pop_size=pop_size, n_iter=n_iter,
                            ms_lower=ms_lower, ms_upper=ms_upper, p_inflate=p_inflate, max_depth=max_depth, init_depth=init_depth, 
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

    # Pickle the results
    if save:
        output_dir = os.path.join(os.getcwd(), "best_params")
        if identifier is None:
            output_file = f"best_slim_{dataset}_{pop_size}_{n_iter}_{scale}.pkl"
        else:
            output_file = f"best_slim_{dataset}_{pop_size}_{n_iter}_{scale}_{identifier}.pkl"
        output_file = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "wb") as f:
            pickle.dump(results_slim, f)
    return results_slim



# -------------------------------- GSGP --------------------------------

def random_search_gsgp(X, y, dataset,scale=False, p_train=0.7, iterations=50, 
                       pop_size=100, n_iter=100, verbose=0, threshold=100000, show_progress=True):
    """
    Perform a random search for the best hyperparameters for the GSGP algorithm.

    Arguments
    ---------
    X: torch.tensor
        The input data.
    y: torch.tensor     
        The target data.
    dataset: str
        The name of the dataset.
    scale: bool
        Whether to scale the data or not.
    p_train: float
        The percentage of the training set.
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
    show_progress: bool
        Whether to show the progress bar or not.

    Returns
    -------
    best_hyperparameters: dict
        A dictionary containing the best hyperparameters for the GSGP algorithm
    """
    
    # Define parameter space
    params = {
        'p_xo': [0, 0],
        'init_depth': [3, 4, 5, 6, 7, 8],
        'prob_const': [0.05, 0.1, 0.15, 0.2],
        'tournament_size': [2, 3, 4],
        'ms_lower': [0, 0, 0, 0.05, 0.1, 0.15],
        'ms_upper': [1, 1, 1, 0.8, 0.6, 0.4, 0.2],
    }

    # Random search loop
    results = {}
    for i in tqdm(range(iterations), disable=not show_progress):
        # Dataset split
        X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train, seed=i)

        if scale:
            scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
            X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
            y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

        # Randomly select parameters
        p_xo = np.random.choice(params['p_xo'])
        init_depth = int(np.random.choice(params['init_depth']))
        prob_const = np.random.choice(params['prob_const'])
        tournament_size = int(np.random.choice(params['tournament_size']))
        ms_lower = np.random.choice(params['ms_lower'])
        ms_upper = np.random.choice(params['ms_upper'])

        # Run GSGP
        try:
            gsgp_model = gsgp(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                pop_size=pop_size, n_iter=n_iter,
                p_xo=p_xo, init_depth=init_depth,
                prob_const=prob_const, tournament_size=tournament_size,
                dataset_name='random_search_gsgp',
                verbose=verbose, log_level=0, minimization=True, reconstruct=True, 
                ms_lower=ms_lower, ms_upper=ms_upper
            )

            if gsgp_model.nodes > threshold:
                # Skip if the tree is too large
                print(f"Parameters: {p_xo}, {init_depth}, {prob_const}, {tournament_size}, {ms_lower}, {ms_upper}")

            # Predict and evaluate
            predictions = gsgp_model.predict(X_test)
            rmse_score = float(rmse(y_true=y_test, y_pred=predictions))

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

    # Pickle the results
    output_dir = os.path.join(os.getcwd(), "best_params")
    output_file = f"best_gsgp_{dataset}_{pop_size}_{n_iter}_{scale}.pkl"
    output_file = os.path.join(output_dir, output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(best_hyperparameters, f)

    return best_hyperparameters



# -------------------------------- GP --------------------------------
def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError('Function call timed out')]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutError('Function call timed out')
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator


@timeout(100)
def run_gp_with_timeout(X_train, y_train, X_test, y_test, pop_size, n_iter, p_xo, max_depth, init_depth, prob_const, dataset_name, verbose):
    gp_model = gp(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        pop_size=pop_size, n_iter=n_iter,
        p_xo=p_xo, max_depth=max_depth,
        init_depth=init_depth, prob_const=prob_const,
        dataset_name=dataset_name,
        verbose=verbose, log_level=0, minimization=True
    )
    return gp_model


def random_search_gp(X, y, dataset, scale=False, p_train=0.7, iterations=50,
                        pop_size=100, n_iter=100, verbose=0, threshold=100000, show_progress=True):
        """
        Perform a random search for the best hyperparameters for the GP algorithm.
    
        Arguments
        ---------
        X: torch.tensor
            The input data.
        y: torch.tensor     
            The target data.
        dataset: str
            The name of the dataset.
        scale: bool
            Whether to scale the data or not.
        p_train: float
            The percentage of the training set.
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
        show_progress: bool
            Whether to show the progress bar or not.
    
        Returns
        -------
        best_hyperparameters: dict
            A dictionary containing the best hyperparameters for the GP algorithm
        """
        
        # Define parameter space
        params = {
            "p_xo" : [0.35, 0.5, 0.6, 0.7, 0.8, 0.9],
            "max_depth" : [14,15,16,17,18,19],
            "init_depth" : [5, 6, 7, 8],
            "prob_const" : [0, 0.05, 0.1, 0.15, 0.2],
        }

        # Random search loop
        results = {}

        for i in tqdm(range(iterations), disable=not show_progress):
            # Dataset split
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train, seed=i)

            if scale:
                scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
                X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
                X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
                y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
                y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

            # Randomly select parameters
            p_xo = np.random.choice(params['p_xo'])
            max_depth = int(np.random.choice(params['max_depth']))
            init_depth = int(np.random.choice(params['init_depth']))
            prob_const = np.random.choice(params['prob_const'])

            # Run GP
            try:
                gp_model = run_gp_with_timeout(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                pop_size=pop_size, n_iter=n_iter,
                p_xo=p_xo, max_depth=max_depth,
                init_depth=init_depth, prob_const=prob_const,
                dataset_name='random_search_gp',
                verbose=verbose
            )

                # Predict and evaluate
                predictions = gp_model.predict(X_test)
                rmse_score = float(rmse(y_true=y_test, y_pred=predictions))

                # Store results
                results[rmse_score] = {
                    'p_xo': p_xo,
                    'max_depth': max_depth,
                    'init_depth': init_depth,
                    'pop_size': pop_size,
                    'n_iter': n_iter,
                    'prob_const': prob_const,
                }

            except TimeoutError:
                print(f"Iteration {i} timed out after 100 seconds")

            except Exception as e:
                print(f"Iteration {i} failed with error: {e}")

        # Sort results by RMSE and return the best configuration
        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}
        best_hyperparameters = list(results.values())[0] if results else {}

        # Pickle the results
        output_dir = os.path.join(os.getcwd(), "best_params")
        output_file = f"best_gp_{dataset}_{pop_size}_{n_iter}_{scale}.pkl"
        output_file = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "wb") as f:
            pickle.dump(best_hyperparameters, f)

        return best_hyperparameters

# -------------------------------- MAIN --------------------------------