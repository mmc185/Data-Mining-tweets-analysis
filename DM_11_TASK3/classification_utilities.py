import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import statistics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
import pandas as pd
from pandas import Index
from utilities import get_path
import os
from sklearn.feature_selection import SelectKBest, chi2, RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def get_metrics(target, pred, target_labels, set_kind, verbose=True):
    """Print accuracy metrics, if verbose is True print also precision, recall, F1 and support scores.

    Parameters
    ----------
    target: array-like of shape (n_samples)
        Data labels.
    
    pred: array-like of shape (n_samples)
        Labels obtained during classification.

    target_labels: array-like of shape (n_classes)
        Names to assign to each class.
    
    set_kind: str
        String indicating which data set has been used (train,validation,test).
    
    verbose: boolean, default='True'
        Boolean used to print more metrics or not.
    """
    print('Accuracy', metrics.accuracy_score(target, pred))
    if verbose:
        print(f'Precision {set_kind} set ', metrics.precision_score(target, pred, average='weighted'))
        print(f'Recall {set_kind} set ', metrics.recall_score(target, pred, average='weighted'))
        print(f'F1 score {set_kind} set ', metrics.f1_score(target, pred, average='weighted'))
        print(f'Support {set_kind} set ', metrics.precision_recall_fscore_support(target, pred))

    print(metrics.classification_report(target, pred, target_names=target_labels))


def confusion_matrix(target, pred, path=None):
    """Compute the confusion matrix for the classification results.

    Parameters
    ----------
    target: array-like of shape (n_samples)
        Data labels.
    
    pred: array-like of shape (n_samples)
        Labels obtained during classification.

    path: str, default='None'
        Path used to select the location where the file needs to be saved. If None it shows the plot without saving it.
    """
    cm = metrics.confusion_matrix(target, pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def cross_validation(model, tr, target):
    """Compute cross validation.

    Parameters
    ----------
    model : estimator object implementing 'fit'
        The object to use to fit the data.
    
    tr: array-like of shape (n_samples, n_features)
        Training set.

    target: array-like of shape (n_samples)
        Training set labels.
    
    Returns
    -------
    cv_scores : dictionary
        Mean of accuracy, recall and f1 scores obtained in the cross validation.
    
    """
    cv_scores = cross_validate(model, tr, target, cv=5, return_train_score=True, scoring=['accuracy', 'recall', 'f1'])

    for k in cv_scores.keys():
        cv_scores[k] = statistics.mean(cv_scores[k])
    return cv_scores


def discretize_column(df, column_name):
    """Convert indicated column of the dataframe from categorical values to discrete values.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe containing columns to convert.
    
    column_name: str
        Name of the column to convert.
    
    """
    # Get the unique variable's values
    values = df[column_name].unique()

    d = dict((el, i) for i, el in enumerate(values))

    df[column_name + '_discr'] = df[column_name].apply(lambda x: d[x])


def prepare_data(scaler=None, k=0):
    """Read from file the dataset and split it into development and test set. Optionally it performs also scaling and feature selection.

    Parameters
    ----------
    scaler : sklearn scaler object, default='None'
        The object to use to scale data. If None no scaling is applied.
    
    k : int, default=0
        Number of feature to select. If 0 no feature selection is applied.
    
    Returns
    -------
    dev : array-like of shape (n_dev_samples, n_features)
        Extracted development set.

    test : array-like of shape (n_test_samples, n_features)
        Extracted test set.

    dev_labels : array-like of shape (n_dev_samples)
        Development set labels.

    test_labels : array-like of shape (n_test_samples)
        Test set labels.
    """
    DATA_PATH = get_path()

    if scaler is not None:
        df_indicators = pd.read_csv(DATA_PATH + 'indicators_clean.csv', sep='#')
        df_users = pd.read_csv(DATA_PATH + 'users_clean.csv', sep='#')

        df_users.id = df_users.id.astype(str)
        df_merge = df_users.merge(df_indicators, left_on='id', right_on='user_id', how='left')
    else:
        df_merge = pd.read_csv(DATA_PATH + "data_scaled_for_clustering.csv", sep='#')

    # Drop variables which aren't useful for classification purposes
    df_merge.drop(columns=['id', 'name', 'user_subscription', 'user_id'], inplace=True)

    # Convert lang column to numbers (e.g. en = 0, it = 1, etc.)
    discretize_column(df_merge, 'lang')

    # Drop original lang column
    df_merge.drop(columns=['lang'], inplace=True)
    # Eliminate NaN values
    df_merge = df_merge.fillna(value=0)
    df_merge.replace(-np.inf, np.finfo(np.float32).min, inplace=True)

    df_merge.replace(np.inf, np.finfo(np.float32).max, inplace=True)

    if scaler is not None:
        df_merge_scaled = scaler.fit_transform(df_merge.values)
        df_merge = pd.DataFrame(df_merge_scaled, columns=df_merge.columns)

    dev, test, dev_labels, test_labels = train_test_split(df_merge.drop(columns='bot'), df_merge['bot'],
                                                          stratify=df_merge['bot'], test_size=0.20)

    if k > 0:
        selector = SelectKBest(chi2, k=k)
        dev_sel = selector.fit_transform(dev, dev_labels)
        test_sel = selector.transform(test)

        print(dev.columns[selector.get_support()])

        return dev_sel, test_sel, dev_labels, test_labels

    return dev, test, dev_labels, test_labels


def grid_search_with_feature_selection(classifier_class, parameters, name, tr, ts, tr_target, ts_target, n_jobs=6,
                                       folds=4, n_features=25):
    """Perform grid search applying three different feature selection approaches (SelectKBest, RFECV, SelectFromModel). Save results on a file and apply testing on the best result.

    Parameters
    ----------
    classifier_class : estimator object implementing 'fit'
        The object to use to fit the data.
    
    parameters : dictionary
        Dictionary containing all the possible values for each hyper-parameter.

    name : str
        Directory name in which to save results.

    tr : pandas.Dataframe
        Data used to apply training.

    ts : pandas.Dataframe
        Data used to test the best model.

    tr_target : array-like of shape (n_train_samples)
        Train set labels.
    
    ts_target : array-like of shape (n_test_samples)
        Test set labels.

    n_jobs : int, default=6
        Number of threads to use during grid search.

    folds : int, default=4
        Number of folds to use in the KFoldCV.

    n_features : int, default=25
        Number of features to select with the SelectKBest feature selection.
    """
    out_path = f'results/{name}/'
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass



    # Different criteria for feature selection
    feature_selection_list = [SelectKBest(chi2, k=n_features),
                              RFECV(estimator=LogisticRegression(max_iter=500), step=1, cv=folds, scoring='accuracy',
                                    min_features_to_select=5),
                              SelectFromModel(estimator=RandomForestClassifier())]

    for fs in feature_selection_list:

        # Apply Feature Selection
        tr_sel = fs.fit_transform(tr, tr_target)
        ts_sel = fs.transform(ts)
        fs_name = str(type(fs).__name__)

        print(f'Chosen columns of {fs_name}:', tr.columns[fs.get_support()])

        # Perform grid search and save best model
        results_df = grid_search(classifier_class, parameters, f"{name}/{fs_name}", tr_sel, tr_target,
                                 n_jobs=n_jobs, k=folds)

        results_df['Feature_Selector'] = [fs_name] * len(results_df)
        results_df['Chosen_columns'] = [tr.columns[fs.get_support()]] * len(results_df)
        results_df.to_csv(f"{out_path}/{fs_name}/gs_results.csv")

        test_best(classifier_class, tr_sel, ts_sel, tr_target, ts_target, out_path=f"{out_path}/{fs_name}/",
                  results_df=results_df)


def test_best(classifier_class, tr, ts, tr_target, ts_target, out_path, results_df=None, in_path=None, sep=','):

    """Evaluate results obtained from a grid search, find the best model w.r.t. accuracy and print its scores.

    Parameters
    ----------
    classifier_class : estimator object implementing 'fit'
        The object to use to fit the data.

    tr : pandas.Dataframe
        Data used to apply training.

    ts : pandas.Dataframe
        Data used to test the best model.

    tr_target : array-like of shape (n_train_samples)
        Train set labels.
    
    ts_target : array-like of shape (n_test_samples)
        Test set labels.

    out_path : str
        Path in which to save the results.

    results_df : int, default='None'
        Dataframe containing the results of a grid search. If None results are taken from a file otherwise an exception is raised.

    in_path : str, default='None'
        Path from which to read the results.If None use the passed results_df otherwise an exception is raised.

    sep : str, default=','
        Number of features to select with the SelectKBest feature selection.
    
    Returns
    -------
    best_classifier : estimator object implementing 'fit'
        The object of the best model found.
    """

    if results_df is None and in_path is not None:
        results_df = pd.read_csv(in_path, sep=sep)

    if results_df is None and in_path is None:
        raise ValueError('results_df and in_path cannot be both None')

    if 'Chosen_columns' in results_df.columns:
        feature_list = eval(results_df.iloc[0]['Chosen_columns']).to_list()
        tr = tr[feature_list]
        ts = ts[feature_list]

    try:
        best_result = results_df.loc[results_df.mean_test_f1.idxmax()][
            ['params', 'mean_train_accuracy', 'mean_train_recall', 'mean_train_precision', 'mean_train_f1',
             'mean_test_accuracy',
             'mean_test_recall', 'mean_test_precision', 'mean_test_f1']]
    except KeyError:
        return None, None

    print(f'Best combo:\n\tparams: {best_result["params"]}'
          f'\n\tmean_train_accuracy: {best_result["mean_train_accuracy"]}'
          f'\n\tmean_train_recall: {best_result["mean_train_recall"]}'
          f'\n\tmean_train_precision: {best_result["mean_train_precision"]}'
          f'\n\tmean_train_f1: {best_result["mean_train_f1"]}'
          f'\n\tmean_val_accuracy: {best_result["mean_test_accuracy"]}'
          f'\n\tmean_val_recall: {best_result["mean_test_recall"]}'
          f'\n\tmean_val_precision: {best_result["mean_test_precision"]}'
          f'\n\tmean_val_f1: {best_result["mean_test_f1"]}\n')
    best_params = eval(best_result['params'])
    #best_params = best_result['params']

    best_classifier = classifier_class(**best_params)
    best_classifier.fit(tr, tr_target)
    ts_pred = best_classifier.predict(ts)
    print("Test set metrics: ")
    get_metrics(ts_target, ts_pred, ['genuine_user', 'bot'], 'test')
    confusion_matrix(ts_target, ts_pred, out_path + 'confusion_matrix.png')
    return best_classifier


def grid_search(classifier_class, parameters, name, tr, tr_target, n_jobs=6, k=4):
    
    """Perform grid search and save results on a file.

    Parameters
    ----------
    classifier_class : estimator object implementing 'fit'
        The object to use to fit the data.
    
    parameters : dictionary
        Dictionary containing all the possible values for each hyper-parameter.

    name : str
        Directory name in which to save results.

    tr : pandas.Dataframe
        Data used to apply training.

    tr_target : array-like of shape (n_train_samples)
        Train set labels.

    n_jobs : int, default=6
        Number of threads to use during grid search.

    k : int, default=4
        Number of folds to use in the KFoldCV.

    Returns
    -------
    results_df : pandas.Dataframe
        Dataframe containing the results of the grid search.    
    """

    out_path = f'results/{name}'
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    gs = GridSearchCV(classifier_class(), param_grid=parameters, scoring=['accuracy', 'precision', 'recall', 'f1'],
                      verbose=3,
                      refit=False, n_jobs=n_jobs, return_train_score=True, cv=k)
    gs.fit(tr, tr_target)
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(f"{out_path}/gs_results.csv")

    return results_df