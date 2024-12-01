from concurrent.futures import ProcessPoolExecutor
from functions.test_algorithms import *
from functions.random_search import random_search_slim  # Assuming this is the non-parallel version
from slim_gsgp_lib.datasets.data_loader import *
import pickle
from tqdm import tqdm

datasets = [globals()[i] for i in globals() if 'load' in i][2:]

# Settings 
pop_size = 100
n_iter = 200
n_iter_rs = 100
p_train = 0.7

def process_dataset(args):
    dataset_loader, scale, struct_mutation, xo, mut_xo = args
    X, y = dataset_loader()
    dataset_name = dataset_loader.__name__

    # Random Search
    results = random_search_slim(
        X, y, dataset_name, scale=scale,
        p_train=p_train, iterations=n_iter_rs, pop_size=pop_size, n_iter=n_iter,
        struct_mutation=struct_mutation, show_progress=False,
        x_o=xo, save=False, mut_xo=mut_xo
    )

    scale_suffix = 'scaled' if scale else 'unscaled'
    xo_suffix = 'xo' if xo else 'no_xo'
    gp_xo_suffix = 'mut_xo' if mut_xo else 'no_mut_xo'
    struct_mutation_suffix = 'struct_mutation' if struct_mutation else 'no_struct_mutation'

    if not os.path.exists('params'):
        os.makedirs('params')
    
    with open(f'params/{dataset_name}_{scale_suffix}_{xo_suffix}_{gp_xo_suffix}_{struct_mutation_suffix}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    # Define tasks for both scaled and unscaled processing                          #Scale, strmt, x_o, mut_xo
    tasks = [(loader, True, False, False, False) for loader in datasets] + [(loader, False, False, False, False) for loader in datasets]  # Scaled + unscaled
    tasks += [(loader, True, True, False, False) for loader in datasets] + [(loader, True, False, True, False) for loader in datasets]    # Strmut + xo
    tasks += [(loader, True, True, True, False) for loader in datasets] + [(loader, True, True, True, True) for loader in datasets]       # Strmut and xo + strmut and xo and mut_xo

    # Parallel processing for both scaled and unscaled datasets
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_dataset, tasks), total=len(tasks)))