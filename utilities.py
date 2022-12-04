import os
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
import seaborn as sn

def get_path():
    DATA_PATH = 'G:/Shared drives/DM_tweets/data/'
    if not os.path.exists(DATA_PATH):
        DATA_PATH = 'G:/Drive condivisi/DM_tweets/data/'
        if not os.path.exists(DATA_PATH):
            os.system('cmd /k"cp ./utilities.py /content/drive/MyDrive/code/utilities.py"')

            from google.colab import drive

            drive.mount('/content/drive')

            DATA_PATH = '/content/drive/Shareddrives/DM_tweets/data/'
            os.system('cmd /k"cp /content/drive/MyDrive/code/utilities.py ."')

    return DATA_PATH

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


# Plot a boxplot w.r.t. a single attribute passed as parameter.
def plot_boxplot(df, col, log=False, path=None):
    df_copy = df.copy()
    df_copy[col] = df_copy[col].fillna(-1.0)

    # Plot the distribution of the indicated column
    plt.title(col)
    plt.boxplot(df_copy[df_copy[col] != -1.0][col], showmeans=True)
    if log:
        plt.yscale('log')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def eval_correlation(df_corr, method='pearson', path=None, figsize=(10,10)):
    correlation_matrix = df_corr.corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)  # Sample figsize in inches
    sn.heatmap(correlation_matrix, annot=True, linewidths=.5, ax=ax)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

def heatmap(matrix, path=None, figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)  # Sample figsize in inches
    sn.heatmap(matrix, annot=True, linewidths=.5, ax=ax)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

#Plot a histogram w.r.t. a single attribute passed as parameter.
def plot_hist(dataframe, attribute_name, log=False, path=None):
    df = pd.DataFrame()

    if log:
        log_attribute_name = attribute_name + '_log'
        df[log_attribute_name] = np.log(dataframe[attribute_name].values)

        attribute_name = log_attribute_name
        df[attribute_name] = df[attribute_name].replace(-np.inf, 0)

    else:
        df[attribute_name] = dataframe[attribute_name].values
    n_bins = math.ceil(np.log2(len(df[attribute_name])) + 1) # Sturges' rule
    df.hist(attribute_name, bins=n_bins, log=True)
    if path is not None:
        plt.savefig(path)


def date_labels(df, attr, format='M', axis='x', n_ticks = 6):
    """
    Utility function to put human readable labels for timestamps instead of seconds.
    :param df:
    :param attr:
    :param format:
    :param axes: 'x','y', the axes where to convert the date labels
    :return:
    """
    # create indices and labels for the plot. This is done because if you pass the entire sequence it takes years to compute
    indices = [df[attr].min() + q*(df[attr].max()-df[attr].min()) for q in np.array(list(range(0,n_ticks)))/(n_ticks-1)]

    if axis == 'x':
        plt.xticks(indices, labels=pd.to_datetime(indices).to_period(format))
    if axis == 'y':
        plt.yticks(indices, labels=pd.to_datetime(indices).to_period(format))

def _scatter_show(df, x_attr, y_attr, path=None, log=False, date_axis=None, xlabel=None, ylabel=None):
    if xlabel is None:
        plt.xlabel(x_attr)
    else:
        plt.xlabel(xlabel)

    if ylabel is None:
        plt.ylabel(y_attr)
    else:
        plt.ylabel(ylabel)

    if date_axis is not None:
        if date_axis == 'both' or date_axis=='x':
            date_labels(df, x_attr, axis='x')
        if date_axis == 'both' or date_axis=='y':
            date_labels(df, y_attr, axis='y')

    if log:
        plt.yscale('log')
    plt.legend()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)

def plot_scatter(df, x_attr, y_attr, path=None, log=False, date_axis=None, xlabel=None, ylabel=None):
    """

    :param df: dataframe containing x_attr, y_attr
    :param x_attr: dataframe attribute to put on the x axis.
    :param y_attr: dataframe attribute to put on the y axis.
    :param path: path where to save the plot. If none just calls plt.show().
    :param log: boolean. Whether to use or not log scale on y axis.
    :param date_axis: None, 'x', 'y', or 'both'. If there is some timestamp dimension, pass the respective axis to convert the ticks in human readable format.
    :return:
    """

    df_copy = df[[x_attr, y_attr]].copy()

    if log:
        df_copy[y_attr] = df_copy[y_attr].replace(0, 0.5)

    plt.scatter(df_copy[x_attr], df_copy[y_attr], color='g', marker='*')
    _scatter_show(df_copy, x_attr, y_attr, log=log, path=path, date_axis=date_axis, xlabel=xlabel, ylabel=ylabel)



def bot_scatter(df, x_attr, y_attr, log=False, path=None, date_axis=None, xlabel=None, ylabel=None):

    df_copy = df[[x_attr, y_attr, 'bot']].copy()

    if log:
        df_copy[y_attr] = df_copy[y_attr].replace(0, 0.5)

    plt.scatter(df_copy[df_copy['bot'] == 0][x_attr],
                df_copy[df_copy['bot'] == 0][y_attr], color='g', marker='*', label='Non-bot user')
    plt.scatter(df_copy[df_copy['bot'] == 1][x_attr],
                df_copy[df_copy['bot'] == 1][y_attr], color='r', marker='2', label='Bot user')

    _scatter_show(df_copy, x_attr, y_attr, log=log, path=path, date_axis=date_axis, xlabel=xlabel, ylabel=ylabel)

def lang_scatter(df, attr, log=False, path=None):
    plt.figure(figsize=(10, 5))
    bot_scatter(df, 'lang', attr, log, path)

