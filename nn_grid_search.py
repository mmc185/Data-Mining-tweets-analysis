from classification_utilities import *
import pandas as dp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

tr, ts, tr_target, ts_target = prepare_data(MinMaxScaler())
print(f'Number of samples in Training set:', len(tr))
print(f'Number of samples in Test set:', len(ts))

parameters = {
    "hidden_layer_sizes": [[5], [10], [16]],
    "activation": ["logistic", "tanh"],
    "solver": ["adam"],
    "alpha": [1e-5, 1e-8],
    "batch_size": [8, 16], #, 32],
    "learning_rate_init": [1e-2, 1e-3, 5e-3],
    "learning_rate": ['constant'],
    "momentum": [0.0, 0.2] #, 0.6] #, 0.0, 0.2, 0.8]
}

#results_df = grid_search(MLPClassifier, parameters, 'nn', tr, tr_target, n_jobs=2)
#best_classifier = test_best(MLPClassifier, tr, ts, tr_target, ts_target, 'classification/nn/', results_df=results_df)
grid_search_with_feature_selection(MLPClassifier, parameters, 'nn', tr, ts, tr_target, ts_target, n_jobs=-1,
                                       folds=4, n_features=25)
