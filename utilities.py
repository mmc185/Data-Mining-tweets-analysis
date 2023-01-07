import os
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
import seaborn as sn
from config import DATA_PATH as config_data_path
TWITTER_FOUNDATION = pd.to_datetime(["20060321"]).astype(np.int64)[0]
SEP_2022 = pd.to_datetime(["20220915"]).astype(np.int64)[0]

def get_path():
    """
    Returns the path from where to load the dataset and where to write the output csv files.
    """
    dev_paths = [config_data_path,
                 'G:/Shared drives/DM_tweets/data/',
                 'G:/Drive condivisi/DM_tweets/data/',
                 '../DM_data/',
                 'H:/Drive condivisi/DM_tweets/data/',
                 '../../../../Data Mining/',
                 ]

    for path in dev_paths:
        if os.path.exists(path):
            return path

    os.system('cmd /k"cp ./utilities.py /content/drive/MyDrive/code/utilities.py"')
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = '/content/drive/Shareddrives/DM_tweets/data/'
    os.system('cmd /k"cp /content/drive/MyDrive/code/utilities.py ."')

    return DATA_PATH

def to_float(x):
    """
    Converts a value to float, setting NaN values to -1. 
     x: any value castable to float.
    :return: the converted value.
    """
    try:
        x = float(x)
        if (np.isnan(x)):
            return float(-1)
        else:
            return x
    except:
        return float(-1)



def select_best_created_at(dates, start_date=TWITTER_FOUNDATION, end_date=SEP_2022):
    """
    Select valid dates from a pandas series of  datetimes.
    Parameters
    ----------
    dates: Pandas series of dates.
    start_date: datetime object. Datetimes before start_date are considered invalid.
    end_date: datetime object. Datetimes after end_date are considered invalid.
    
    Returns
    -------
    An empty list if no date is valid. 
    A datetime if there is only one valid date. 
    A list of datetime objects if there is more than one valid date.
    """
    correct = []
    wrong = []

    for date in dates:
        if date < end_date and date >= start_date:
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

# 
def plot_boxplot(df, attribute, log=False, out_path=None, date=False, timedelta=False):
    """
    Plot a boxplot w.r.t. a single attribute passed as parameter.
    
    Parameters
    ----------
        df: the dataframe
        attribute: string. The attribute wrt which the boxplot is computer.
        log: boolean. Whether to scale df[col] in logarithmic scale or not.
        out_path: string. File path to which save the image. If None, the plot is just shown via plt.show().
        date: boolean. If the attribute is a int timestamp (number of seconds), the y-ticks are converted to YYYY-MM format.
        timedelta: boolean. If the attribute is a int timedelta (number of seconds), the y-ticks are converted to YYYY-MM format.
    """
    df_copy = df.copy()
    df_copy[attribute] = df_copy[attribute].fillna(-1.0)

    # Plot the distribution of the indicated column
    plt.title(attribute)
    plt.boxplot(df_copy[df_copy[attribute] != -1.0][attribute], showmeans=True)
    if log:
        plt.yscale('log')

    if date:
        date_labels(df, attribute, axis='y')

    if timedelta:
        timedelta_labels(df, attribute, axis='y')


    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


def eval_correlation(df, method='pearson', out_path=None, figsize=(10, 10)):
    """
    Plots a correlation matrix of a dataframe attributes.
    Parameters
    ----------    
        df: pandas dataframe object.
        method: string. Correlation method supported by pandas.Dataframe.corr()
        out_path: string. File path to which save the image. If None, the plot is just shown via plt.show().
        figsize: tuple. Pyplot figsize parameter.
    """
    correlation_matrix = df.corr(method=method)
    heatmap(correlation_matrix, out_path, figsize)

def heatmap(matrix, out_path=None, figsize=(10, 10)):
    """
    Plots a correlation matrix of a dataframe attributes.
    Parameters
    ----------
        matrix: rectangular dataset
            2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
            is provided, the index/column information will be used to label the
            columns and rows.
        out_path: string. File path to which save the image. If None, the plot is just shown via plt.show().
        figsize: tuple. Pyplot figsize parameter.
    """
    fig, ax = plt.subplots(figsize=figsize)  # Sample figsize in inches
    sn.heatmap(matrix, annot=True, linewidths=.5, ax=ax)
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
def plot_hist(df, attribute, log=False, out_path=None, date=False):
    """
    Plot a histogram w.r.t. a single attribute passed as parameter.

    Parameters
    ----------
        df: the dataframe
        attribute: string. The attribute wrt which the boxplot is computer.
        log: boolean. Whether to scale df[col] in logarithmic scale or not.
        out_path: string. File path to which save the image. If None, the plot is just shown via plt.show().
        date: boolean. If the attribute is a int timestamp (number of seconds), the x-ticks are converted to YYYY-MM format.
    """
    transformed_df = pd.DataFrame()

    if log:
        log_attribute_name = attribute + '_log'
        transformed_df[log_attribute_name] = np.log(df[attribute].values)

        attribute = log_attribute_name
        transformed_df[attribute] = transformed_df[attribute].replace(-np.inf, 0)

    else:
        transformed_df[attribute] = df[attribute].values
    n_bins = math.ceil(np.log2(len(transformed_df[attribute])) + 1) # Sturges' rule
    transformed_df.hist(attribute, bins=n_bins, log=True)

    if date:
        x_ticks = plt.xticks()
        plt.xticks(x_ticks[0], pd.to_datetime(x_ticks[0]).to_period('M'))

    if out_path is not None:
        plt.savefig(out_path)


def date_labels(df, attribute, format='M', axis='x', n_ticks = 6):
    """
    Utility function to put human readable labels for timestamps instead of seconds.

    Parameters
    ----------
        df: pandas.Dataframe containing attribute.
        attribute: string. The df attribute to convert.
        format: string. The format to which convert the datetimes. Default is 'M', which converts to 'YYYY-MM'.
        axis: string. Accepted values are 'x' and'y'. The axes where to convert the date labels. Default is 'x'.
        n_ticks: how many ticks to display on the selected axis. Default is 6.
    :return:
    """
    # create indices and labels for the plot. This is done because if you pass the entire sequence it takes years to compute
    indices = [df[attribute].min() + q * (df[attribute].max() - df[attribute].min()) for q in np.array(list(range(0, n_ticks))) / (n_ticks - 1)]
    labels = pd.to_datetime(indices).to_period(format)
    if axis == 'x':
        plt.xticks(indices, labels= labels)
    if axis == 'y':
        plt.yticks(indices, labels= labels)

def timedelta_labels(df, attribute, axis='x', n_ticks = 6):
    """
    Utility function to put human readable labels for timedeltas instead of seconds. The seconds are converted in days.

    Parameters
    ----------
        df: pandas.Dataframe containing attribute.
        attribute: string. The df attribute to convert.
        axis: string. Accepted values are 'x' and'y'. The axes where to convert the date labels. Default is 'x'.
        n_ticks: how many ticks to display on the selected axis. Default is 6.
    """

    # create indices and labels for the plot. This is done because if you pass the entire sequence it takes years to compute
    indices = [df[attribute].min() + q * (df[attribute].max() - df[attribute].min()) for q in np.array(list(range(0, n_ticks))) / (n_ticks - 1)]

    labels = pd.to_timedelta(indices).days
    axis_label = "days"
    if axis == 'x':
        plt.xlabel(axis_label)
        plt.xticks(indices, labels=labels)
    if axis == 'y':
        plt.ylabel(axis_label)
        plt.yticks(indices, labels=labels)

def _scatter_show(df, x_attr, y_attr, out_path=None, log=False, date_axis=None, xlabel=None, ylabel=None):
    """
    Creates a scatter and displays and saves it to the specified out_path.
    Parameters
    ----------
        df: pandas.Dataframe containing x_attr, y_attr
        x_attr: string. Dataframe attribute to put on the x axis.
        y_attr: string. Dataframe attribute to put on the y axis.
        out_path: string. Path where to save the plot. If none just calls plt.show().
        log: boolean. Whether to use or not log scale on y axis.
        date_axis: string. Accepted values are None, 'x', 'y', or 'both'. If there is some timestamp dimension, the specified axis ticks are converted to 'YYYY-MM' format..
        xlabel: string. Label for x axis.
        ylabel: string. Label for y axis.
    """
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
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)

def plot_scatter(df, x_attr, y_attr, out_path=None, log=False, date_axis=None, xlabel=None, ylabel=None):
    """
    Plot a scatterplot over two dataframe attributes.

    Parameters
    ----------
     df: pandas.Dataframe containing x_attr, y_attr
     x_attr: string. Dataframe attribute to put on the x axis.
     y_attr: string. Dataframe attribute to put on the y axis.
     out_path: string. Path where to save the plot. If none just calls plt.show().
     log: boolean. Whether to use or not log scale on y axis.
     date_axis: string. Accepted values are None, 'x', 'y', or 'both'. If there is some timestamp dimension, the specified axis ticks are converted to 'YYYY-MM' format..
     xlabel: string. Label for x axis.
     ylabel: string. Label for y axis.
    """

    df_copy = df[[x_attr, y_attr]].copy()

    if log:
        df_copy[y_attr] = df_copy[y_attr].replace(0, 0.5)

    plt.scatter(df_copy[x_attr], df_copy[y_attr], color='g', marker='*')
    _scatter_show(df_copy, x_attr, y_attr, log=log, out_path=out_path, date_axis=date_axis, xlabel=xlabel, ylabel=ylabel)



def bot_scatter(df, x_attr, y_attr, log=False, out_path=None, date_axis=None, xlabel=None, ylabel=None):
    """
    Plot a scatterplot over two dataframe attributes, drawing the points with bot=1 to red and bot=0 in green.

    Parameters
    ----------
        df: pandas.Dataframe containing x_attr, y_attr, and a boolean column named "bot".
        x_attr: string. Dataframe attribute to put on the x axis.
        y_attr: string. Dataframe attribute to put on the y axis.
        log: boolean. Whether to use or not log scale on y axis.
        out_path: string. Path where to save the plot. If none just calls plt.show().
        date_axis: string. Accepted values are None, 'x', 'y', or 'both'. If there is some timestamp dimension, the specified axis ticks are converted to 'YYYY-MM' format..
        xlabel: string. Label for x axis.
        ylabel: string. Label for y axis.
    :return:
    """

    df_copy = df[[x_attr, y_attr, 'bot']].copy()

    if log:
        df_copy[y_attr] = df_copy[y_attr].replace(0, 0.5)

    plt.scatter(df_copy[df_copy['bot'] == 0][x_attr],
                df_copy[df_copy['bot'] == 0][y_attr], color='g', marker='*', label='Non-bot user')
    plt.scatter(df_copy[df_copy['bot'] == 1][x_attr],
                df_copy[df_copy['bot'] == 1][y_attr], color='r', marker='2', label='Bot user', alpha=0.5)

    _scatter_show(df_copy, x_attr, y_attr, log=log, out_path=out_path, date_axis=date_axis, xlabel=xlabel, ylabel=ylabel)

def lang_scatter(df, attr, log=False, out_path=None):
    """
    Plots a scatterplot with df["lang"] on the x axis and df["attr"] on the y axis.

    Parameters
    ----------
        df: pandas.Dataframe containing x_attr, y_attr, and a boolean column named "bot".
        attr: string. Dataframe attribute to put on the x axis.
        log: boolean. Whether to use or not log scale on y axis.
        out_path: string. Path where to save the plot. If none just calls plt.show().
    """
    plt.figure(figsize=(10, 5))
    bot_scatter(df, 'lang', attr, log, out_path)

