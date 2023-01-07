from sklearn import metrics
from utilities import *


def get_metrics(data_scaled, labels, print_out=True):
    """Compute Silhouette and Davies Bouldin scores for clustering

    Parameters
    ----------
    data_scaled: pandas.Dataframe
        Data on which clustering was performed

    labels: array-like of shape (n_samples)
        labels obtained during clustering

    print_out: boolean, default='False'
        Boolean value to print scores

    Returns
    -------
    silhouette : float
        Silhouette score
    DBscore : float
        Davies Bouldin score
    """
    silohuette = metrics.silhouette_score(data_scaled, labels)
    DBscore = metrics.davies_bouldin_score(data_scaled, labels)
    if print_out:
        print(f"Silouhette score: {silohuette}")
        print(f"Davies Bouldin score: {DBscore}")
    else:
        return silohuette, DBscore


def sse(df, labels):
    """Compute sse on computed labels

    Parameters
    ----------
    df: pandas.Dataframe
        Data on which clustering was performed

    labels: array-like of shape (n_samples)
        labels obtained during clustering

    Returns
    -------
    sum : float
        SSE score
    """
    sum = 0
    for label in np.unique(labels):
        mean = df[labels==label].mean().values
        row_sums = df.apply(lambda row: np.linalg.norm(row.values - mean)**2,axis=1).sum()
        print(row_sums)
        sum += np.sum(row_sums)
    return sum

def scatterplot(df, attr1, attr2, c_labels, centroids=None, filename=None, filter=None, figsize=(8,8)):
    """Plot scatter chart w.r.t. attr1 and attr2

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe to use to plot

    attr1: str
        Name of the first attribute to plot

    attr2: str
        Name of the second attribute to plot

    labels: array-like of shape (n_samples)
        Labels obtained during clustering

    centroids: array-like of shape (n_samples, n_features)
        Centroids found during clustering
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(df[attr1], df[attr2], c=c_labels, s=10)

    if filter:
        df = df.loc[filter]
        c_labels = c_labels[filter]

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='.', c='r')

    plt.tick_params(axis='both', which='major')
    plt.legend(*scatter.legend_elements())
    plt.savefig(filename)

def plots(df, labels, path=None, centroids=None, attributes=None):
    """Plot different types of charts (landscape, radar, histogram w.r.t. languages, histogram w.r.t. bots) w.r.t. attr1 and attr2

    Parameters
    ----------
    df: pandas.Dataframe
        Data on which clustering was performed

    labels: array-like of shape (n_samples)
        labels obtained during clustering

    Returns
    -------
    sum : float
        SSE score
    """
    if attributes is None:
        attributes = df.columns

    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    # Line plot
    plt.figure(figsize=(40, 4))

    for label in np.unique(labels):
        if centroids is None:
            cent = df[attributes][labels == label].median()
        else:
            cent = centroids[label]
        plt.plot(cent, label="Cluster %s" % label)
    plt.tick_params(axis='both', which='major')
    plt.xticks(range(0, len(df[attributes].columns)), df[attributes].columns, rotation=90)
    plt.legend()
    if path is not None:
        plt.savefig(path + "/landscape.png")
    else:
        plt.show()

    # Radar plot
    N = len(attributes)

    plt.figure(figsize=(8, 8))
    for label in np.unique(labels):
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        if centroids is None:
            values = df[attributes].loc[np.where(labels == label)].median().tolist()
        else:
            values = centroids[label].tolist()
        values += values[:1]
        angles += angles[:1]

        ax = plt.subplot(polar=True)
        plt.xticks(angles[:-1], attributes)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, 'b', alpha=0.1)

    plt.savefig(path + "/radar.png")

    # Plot w.r.t. languages
    lang_ct = pd.crosstab(labels, df['lang'])
    to_agg = lang_ct.columns[lang_ct.sum() < 30]
    lang_ct['others'] = lang_ct[to_agg].sum(axis=1)
    lang_ct.drop(columns = to_agg, inplace=True)
    fig, ax = plt.subplots(figsize=(24, 8))
    lang_ct.plot(kind='bar', stacked=False, ax=ax)
    plt.xlabel('Cluster')
    plt.ylabel('lang')
    plt.yscale('log')
    plt.legend(prop={'size': 15}, loc="upper right")
    plt.savefig(path + "/lang_characterization.png")

    # Plot w.r.t. bots
    bot_ct = pd.crosstab(labels, df['bot'])

    fig, ax = plt.subplots(figsize=(24, 8))  # Sample figsize in inches
    #plt.figure(figsize=(10,25))
    bot_ct.plot(kind='bar', stacked=False, ax=ax)
    plt.xlabel('Cluster')
    plt.ylabel('bot')
    plt.legend(prop={'size': 15}, loc="upper right",  labels=['genuine user','bot'])
    plt.savefig(path + "/bot_characterization.png")


def plot_landscape(labels, centroids, df, attributes=None, path=None, figsize=(40,4)):
    #Landscape plot
    if attributes is None:
        attributes = df.columns
    plt.figure(figsize=figsize)
    for label in np.unique(labels):
        if centroids is None:
            cent = df[attributes][labels == label].median()
        else:
            cent = centroids[label]
        plt.plot(cent, label="Cluster %s" % label)
    plt.tick_params(axis='both', which='major')
    plt.xticks(range(0, len(df[attributes].columns)), df[attributes].columns, rotation=90)
    plt.legend()
    if path is not None:
        plt.savefig(path + "/landscape.png")
    else:
        plt.show()