from concurrent.futures import ProcessPoolExecutor
from functions.test_algorithms import *
from functions.random_search import random_search_slim  # Assuming this is the non-parallel version
from slim_gsgp_lib.datasets.data_loader import *
import pickle
from tqdm import tqdm

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

# Settings 
pop_size = 10
n_iter = 20
n_iter_rs = 10
p_train = 0.7

def process_dataset(args):
    dataset_loader, scale = args
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__

    # Random Search
    results = random_search_slim(
        X, y, dataset_name, scale=scale,
        p_train=p_train, iterations=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
        struct_mutation=False, show_progress=False,
        x_o=False, save=False,
    )

    scale_suffix = 'scaled' if scale else 'unscaled'
    with open(f'params/{dataset_name}_{scale_suffix}.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    # Define tasks for both scaled and unscaled processing
    tasks = [(loader, True) for loader in datasets] + [(loader, False) for loader in datasets]

    # Parallel processing for both scaled and unscaled datasets
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_dataset, tasks), total=len(tasks)))