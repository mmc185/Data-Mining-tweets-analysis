from sklearn import metrics
from utilities import *


def get_metrics(data_scaled, labels, print_out=True):
    silohuette = metrics.silhouette_score(data_scaled, labels)
    DBscore = metrics.davies_bouldin_score(data_scaled, labels)
    if print_out:
        print(f"Silouhette score: {silohuette}")  # [-1, 1] Good when near 1
        print(f"Davies Bouldin score: {DBscore}")  # Good when near 0
    else:
        return silohuette, DBscore

def sse(df, labels):
    sum = 0
    for label in np.unique(labels):
        mean = df[labels==label].mean().values
        row_sums = df.apply(lambda row: np.linalg.norm(row.values - mean)**2,axis=1).sum()
        print(row_sums)
        sum += np.sum(row_sums)
    return sum

def scatterplot(df, attr1, attr2, c_labels, centroids=None, filename=None, filter=None, figsize=(8,8)):
    plt.figure(figsize=figsize)
    #cent = scaler.inverse_transform(kmeans.cluster_centers_)
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
