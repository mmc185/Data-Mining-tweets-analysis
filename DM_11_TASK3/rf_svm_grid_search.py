from classification_utilities import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

tr, ts, tr_target, ts_target = prepare_data(MinMaxScaler())

n_jobs = 6
print(f'Number of samples in Training set:', len(tr))
print(f'Number of samples in Test set:', len(ts))

# Random Forest

parameters = {
    "n_estimators": range(50,500,100),
    "criterion": ["gini","entropy","log_loss"],
    "max_depth": [None] + list(range(5,100,10)),
    "min_samples_split": [2,4,8,16,32,64],
    "min_samples_leaf": [2,4,8,16,32,64,128],
    "max_features":["auto","sqrt","log2",None],
    "max_leaf_nodes":[None] + list(range(1,10,2)),
    "min_impurity_decrease": 0.1 * np.array(range(1,5))
}

results_df = grid_search(RandomForestClassifier, parameters, 'random_forest', tr, ts, tr_target, ts_target,
                             n_jobs=n_jobs)
grid_search_with_feature_selection(RandomForestClassifier, parameters, 'random_forest', tr, ts, tr_target, ts_target,
                                   n_jobs=n_jobs, folds=4, n_features=25)

# SVM

parameters_linear = {
    'C': [10**(exp) for exp in [-2,-1,1,2,3,4]],
    'kernel': ['linear'],
    'random_state': [42]
}

gamma_range = ['scale','auto'] + [10**(exp) for exp in [-2,-1,0]]

parameters_poly = {
    **parameters_linear,
    'kernel': ['poly'],
    'degree': list(range(2,6)),
    'gamma': gamma_range,
}

parameters_rbf = {
    **parameters_linear,
    'kernel': ['rbf'],
    'gamma': gamma_range
}

grid_search(SVC, [parameters_linear,parameters_poly,parameters_rbf], 'svm_all_kernels', tr, ts, tr_target, ts_target, n_jobs=n_jobs)
grid_search_with_feature_selection(SVC, [parameters_rbf, parameters_linear, parameters_poly], 'svm_all_kernels', tr, ts, tr_target, ts_target, n_jobs=n_jobs, folds=4, n_features=15)
