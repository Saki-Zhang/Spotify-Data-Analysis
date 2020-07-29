import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# gets the top 10 distinct artists
def get_top_artists(df):
    artists = df[['Artist', 'Streams']]
    artists = artists.groupby('Artist')
    artists = artists.sum().sort_values('Streams', ascending=False)
    artists = artists.reset_index()
    artists.index = np.arange(1, len(artists) + 1)
    artists = artists.head(10)        # top 10

    return artists

# gets the top 100 distinct tracks
def get_top_tracks(df):
    tracks = df[['Track Name', 'Streams']]
    tracks = tracks.groupby('Track Name')
    tracks = tracks.sum().sort_values('Streams', ascending=False)
    tracks = tracks.reset_index()
    tracks.index = np.arange(1, len(tracks) + 1)
    tracks = tracks.head(100)        # top 100

    return tracks

# gets the top features by sorting according to the number of streams
def get_top_features(featuresdf):
    top_features = featuresdf.sort_values('Streams', ascending=False)

    return top_features

# gets the correlation matrix graph
def get_correlation(top_features):
    features = top_features[top_features.columns[-14:]]
    features = features.drop('Streams', axis=1)

    plt.figure(figsize = (16,5))
    sns.heatmap(features.corr(), cmap='coolwarm', annot=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.set_window_title('Correlations')
    plt.savefig('images/Correlations.png')  # save graph
    plt.show()

# gets the trend graph for each feature
def get_trends(top_features):
    # get top 50 features
    n = 50
    palette = plt.get_cmap('Set1')
    trends_6 = pd.DataFrame({'x': range(1, n + 1),
                     'danceability': top_features['danceability'].head(n),
                     'energy': top_features['energy'].head(n),
                     'speechiness': top_features['speechiness'].head(n),
                     'acousticness': top_features['acousticness'].head(n),
                     'liveness': top_features['liveness'].head(n),
                     'valence': top_features['valence'].head(n)})

    num = 0
    for column in trends_6.drop('x', axis=1):
        num += 1

        plt.subplot(3, 3, num)

        plt.plot(trends_6['x'], trends_6[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)

        # limits x-axis values to top 30 for graphical reasons
        plt.xlim(0, 30)

        # since all values in range 0 to 1 inclusive, set y to [0:1]
        plt.ylim(0, 1)

        # no x labels for graphical reasons and since we only want to see the trends, so the x values do not matter
        plt.tick_params(labelbottom='off')

        # since all of the graphs have the y value ranges, only display them for the left-most graphs
        if num not in [1, 4, 7]:
            plt.tick_params(labelleft='off')

        plt.title(column, loc='left', fontsize=8, fontweight=0, color=palette(num))

    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.set_window_title('Trends')
    plt.savefig('images/Trends.png')        # save graphs
    plt.show()

    trends = pd.DataFrame({'x': range(1, n + 1),
                           'loudness': top_features['loudness'].head(n),
                           'instrumentalness': top_features['instrumentalness'].head(n),
                           'key': top_features['key'].head(n),
                           'mode': top_features['mode'].head(n),
                           'tempo': top_features['tempo'].head(n),
                           'time_signature': top_features['time_signature'].head(n),
                           'duration_ms': top_features['duration_ms'].head(n)})
    num = 0
    for column in trends.drop('x', axis=1):
        num += 1

        plt.subplot(3, 3, num)

        plt.plot(trends['x'], trends[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)

        # no x labels for graphical reasons and since we only want to see the trends, so the x values do not matter
        plt.tick_params(labelbottom='off')

        plt.title(column, loc='left', fontsize=8, fontweight=0, color=palette(num))

    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.set_window_title('Trends')
    plt.savefig('images/Trends2.png')       # save graphs
    plt.show()


df = pd.read_csv('data_us.csv')
featuresdf = pd.read_csv('datasetwithstreams.csv')

top_artists = get_top_artists(df)
print(top_artists)

top_tracks = get_top_tracks(df)
print(top_tracks)

top_features = get_top_features(featuresdf)

get_correlation(top_features)

get_trends(top_features)