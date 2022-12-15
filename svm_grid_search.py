
from classification_utilities import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC

n_jobs = 6
tr, ts, tr_target, ts_target = prepare_data(scaler=MinMaxScaler())
print(f'Number of samples in Training set:', len(tr))
print(f'Number of samples in Test set:', len(ts))

parameters_linear = {
    'C': [10**(exp) for exp in [-2,-1,0,1,2,3,4]],
    'kernel': ['linear'],
    'shrinking': [True, False],
    'tol': [1e-8, 1e-4, 1e-2, 1e-1],
    'random_state': [42]
}

gamma_range = ['scale','auto'] + [10**(exp) for exp in [-3,-2,-1,0,1,2]]

parameters_poly = {
    **parameters_linear,
    'kernel': ['poly'],
    'degree': list(range(2,9)),
    'gamma': gamma_range,
}

parameters_rbf_sigmoid = {
    **parameters_linear,
    'kernel': ['rbf','sigmoid'],
    'gamma': gamma_range
}

grid_search(SVC, [parameters_linear], 'svm', tr, ts, tr_target, ts_target, n_jobs=n_jobs)