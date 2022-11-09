import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
import seaborn as sn


def to_float(x):
    try:
        x = float(x)
        if (np.isnan(x)):
            return float(-1)
        else:
            return x
    except:
        return float(-1)


def select_best_created_at(dates):
    twitter_foundation = pd.to_datetime(["20060321"]).astype(np.int64)[0]
    sep_2022 = pd.to_datetime(["20220915"]).astype(np.int64)[0]

    correct = []
    wrong = []

    for date in dates:
        if date < sep_2022 and date >= twitter_foundation:
            correct.append(date)
        else:
            wrong.append(date)

    if len(correct) == 0:
        # we have no correct candidate, so do not perform any substitution. We will deal with this record in the data cleaning step.
        return []
    if len(correct) == 1:
        # we have only one correct candidate, so we can merge the duplicates into one.
        return correct[0]
    if len(correct) > 1:
        # we return all the candidates.
        return correct


def len_with_int(x):
    try:
        return len(x)
    except TypeError:
        return 1


def plot_hist(dataframe, attribute_name, log=False):
    df = pd.DataFrame()

    if log:
        log_attribute_name = attribute_name + '_log'
        df[log_attribute_name] = np.log(dataframe[attribute_name].values)

        attribute_name = log_attribute_name
        df[attribute_name] = df[attribute_name].replace(-np.inf, 0)

    else:
        df[attribute_name] = dataframe[attribute_name].values
    n_bins = math.ceil(np.log2(len(df[attribute_name])) + 1)  # Sturges' rule
    df.hist(attribute_name, bins=n_bins, log=True)


# Plot a boxplot w.r.t. a single attribute passed as parameter.
def plot_boxplot(df, col, log=False):
    # Plot the distribution of the indicated column
    plt.boxplot(df[df[col] != -1.0][col], showmeans=True)
    if log:
        plt.yscale('log')
    plt.show()


def eval_correlation(df_corr, method='pearson'):
    correlation_matrix = df_corr.corr(method=method)
    fig, ax = plt.subplots(figsize=(10, 10))  # Sample figsize in inches
    sn.heatmap(correlation_matrix, annot=True, linewidths=.5, ax=ax)
    plt.show()

#Plot a histogram w.r.t. a single attribute passed as parameter.
def plot_hist(dataframe, attribute_name, log=False):
    df = pd.DataFrame()

    if log:
        log_attribute_name = attribute_name+'_log'
        df[log_attribute_name] = np.log(dataframe[attribute_name].values)

        attribute_name = log_attribute_name
        df[attribute_name] = df[attribute_name].replace(-np.inf, 0)

    else:
        df[attribute_name] = dataframe[attribute_name].values
    n_bins = math.ceil(np.log2(len(df[attribute_name])) + 1) #Sturges' rule
    df.hist(attribute_name, bins = n_bins, log=True)
