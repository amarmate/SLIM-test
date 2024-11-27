from slim_gsgp_lib.main_slim import slim
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


# ----------------------------------- SLIM ----------------------------------- #
def test_slim(X, y, args_dict=None,
            dataset_name='dataset_1', 
            pop_size=100, 
            n_iter=100, 
            ms_lower=0, 
            ms_upper=1, 
            n_elites=1,
            initializer='rhh',
            iterations=30,
            struct_mutation=False,
            scale=True,
            algorithm="SLIM+SIG1",
            verbose=0,
            p_train=0.7,
            tournament_size=2,
):    
    """

    Arguments
    ---------
    X: torch.tensor
        The input data.
    y: torch.tensor
        The target data.
    args_dict: dict
        A dictionary containing the hyperparameters for the SLIM algorithm.
    dataset_name: str
        The name of the dataset.
    pop_size: int
        The population size.
    n_iter: int     
        The number of iterations to perform.
    ms_lower: float
        The lower bound for the mutation strength.
    ms_upper: float
        The upper bound for the mutation strength.
    n_elites: int
        The number of elites.
    initializer: str
        The initializer to use.
    iterations: int
        The number of iterations to perform.
    struct_mutation: bool
        Whether to use structural mutation or not.
    scale: bool
        Whether to scale the data or not.
    algorithm: str
        The SLIM algorithm to use.
    verbose: int
        The verbosity level.
    p_train: float
        The percentage of the training set.
    tournament_size: int
        The tournament size.

    Returns
    -------
    rmse_: list
        A list containing the RMSE scores.
    mape_: list
        A list containing the MAPE scores.
    nrmse_: list
        A list containing the NRMSE scores.
    r2_: list   
        A list containing the R-squared scores.
    mae_: list  
        A list containing the MAE scores.
    std_rmse_: list 
        A list containing the standardized RMSE scores.
    time_stats: list
        A list containing the time taken to train the model.
    train_fit: list
        A list containing the training fitness scores.
    test_fit: list
        A list containing the test fitness scores.
    size: list
        A list containing the size of the trees.
    """
    
    rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size = [], [], [], [], [], [], [], [], [], []

    for it in tqdm(range(iterations)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train, seed=it)

        if scale:
            scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
            X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
            y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

        algorithm_name = 'MUL-' + algorithm.split('*')[1] if '*' in algorithm else 'ADD-' + algorithm.split('+')[1]
        path = f"logs/{dataset_name}/{algorithm_name}_{it}.log"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        start = time.time()
        final_tree = slim(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                            dataset_name=dataset_name, slim_version=algorithm, pop_size=pop_size, n_iter=n_iter, seed=it, ms_lower=ms_lower, ms_upper=ms_upper,
                            reconstruct=True, n_jobs=1, tournament_size=tournament_size, initializer=initializer, struct_mutation=struct_mutation,
                            log_path=path, verbose=verbose, log_level=3, n_elites=n_elites, **args_dict)
        end = time.time()
        
        # Get the node count of the tree
        nodes_count = final_tree.nodes_count
        time_taken = end - start
        train_fitness_elite = final_tree.fitness.item()
        test_fitness_elite = final_tree.test_fitness.item()

        # Calculate predictions and metrics
        y_pred = final_tree.predict(X_test)
        rmse_score = rmse(y_test, y_pred).item()
        mape_score = mape(y_test, y_pred)
        nrmse_score = nrmse(y_test, y_pred)
        r2_score = r_squared(y_test, y_pred)
        mae_score = mae(y_test, y_pred)
        std_rmse_score = standardized_rmse(y_test, y_pred)
        
        # Append metrics to respective lists
        rmse_.append(rmse_score)
        mape_.append(mape_score)
        nrmse_.append(nrmse_score)
        r2_.append(r2_score)
        mae_.append(mae_score)
        std_rmse_.append(std_rmse_score)
        time_stats.append(time_taken)
        train_fit.append(train_fitness_elite)
        test_fit.append(test_fitness_elite)
        size.append(nodes_count)

    return rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size


# ----------------------------------- GSGP ----------------------------------- #

def test_gsgp(X, y, args_dict=None,
              dataset_name='dataset_1', 
              iterations=30,
              scale=True,
              verbose=0,
              p_train=0.7,
              threshold=100000,
):    
    """
    Arguments
    ---------
    X: torch.tensor
        The input data.
    y: torch.tensor
        The target data.
    args_dict: dict
        A dictionary containing the hyperparameters for the GSGP algorithm.
    dataset_name: str
        The name of the dataset.
    iterations: int
        The number of iterations to perform.
    scale: bool
        Whether to scale the data or not.
    verbose: int
        The verbosity level.
    p_train: float
        The percentage of the training set.
    threshold: int
        The maximum number of nodes allowed in the tree.


    Returns
    -------
    rmse: list
        A list containing the RMSE scores.
    mape: list
        A list containing the MAPE scores.
    nrmse: list
        A list containing the NRMSE scores.
    r2: list   
        A list containing the R-squared scores.
    mae: list  
        A list containing the MAE scores.
    std_rmse: list 
        A list containing the standardized RMSE scores.
    time_stats: list
        A list containing the time taken to train the model.
    train_fit: list
        A list containing the training fitness scores.
    test_fit: list
        A list containing the test fitness scores.
    size: list
        A list containing the size of the trees.
    """
    
    rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size = [], [], [], [], [], [], [], [], [], []

    for it in tqdm(range(iterations)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=1-p_train, seed=it)

        if scale:
            scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
            X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
            y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1), dtype=torch.float32)

        path = f"logs/{dataset_name}/GSGP_{it}.log"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        start = time.time()
        final_tree = gsgp(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          dataset_name=dataset_name, seed=it,
                          log_path=path, verbose=verbose, **args_dict, reconstruct=True)
        end = time.time()

        if final_tree.nodes > threshold:
            print(f"Tree too large: {final_tree.nodes}")
            continue
        
        # Get the node count of the tree
        nodes_count = final_tree.nodes
        time_taken = end - start
        train_fitness_elite = final_tree.fitness.item()
        test_fitness_elite = final_tree.test_fitness.item()

        # Calculate predictions and metrics
        y_pred = final_tree.predict(X_test)
            
        rmse_score = rmse(y_test, y_pred).item()
        mape_score = mape(y_test, y_pred)
        nrmse_score = nrmse(y_test, y_pred)
        r2_score = r_squared(y_test, y_pred)
        mae_score = mae(y_test, y_pred)
        std_rmse_score = standardized_rmse(y_test, y_pred)
        
        # Append metrics to respective lists
        rmse_.append(rmse_score)
        mape_.append(mape_score)
        nrmse_.append(nrmse_score)
        r2_.append(r2_score)
        mae_.append(mae_score)
        std_rmse_.append(std_rmse_score)
        time_stats.append(time_taken)
        train_fit.append(train_fitness_elite)
        test_fit.append(test_fitness_elite)
        size.append(nodes_count)

    return rmse_, mape_, nrmse_, r2_, mae_, std_rmse_, time_stats, train_fit, test_fit, size