from classification_utilities import *
import pandas as dp
from sklearn.neural_network import MLPClassifier

tr, ts, tr_target, ts_target = prepare_data()
print(f'Number of samples in Training set:', len(tr))
print(f'Number of samples in Test set:', len(ts))

parameters = {
    "hidden_layer_sizes": [[10]], # , [5], [16]],
    "activation": ["tanh"], #, "tanh"],
    "solver": ["adam"],
    "alpha": [1e-5, 1e-8],
    "batch_size": [8, 16, 32],
    "learning_rate_init": [1e-2, 1e-3, 5e-3],
    "learning_rate": ['adaptive'],  # 'costant'],
    "momentum": [0.0, 0.2]  # 0.4, 0.6, 0.8]
}

nn, results_df = grid_search(MLPClassifier, parameters, 'nn', tr, ts, tr_target, ts_target, n_jobs=2)
