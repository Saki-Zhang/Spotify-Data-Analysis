import requests
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/saki/Downloads/data.csv')
df['ID'] = df['URL'].str.replace('https://open.spotify.com/track/', '')

data_us = df[df.Region == 'us'].drop(['URL', 'Region'], axis=1)
data_us.to_csv('data_us.csv')

# print(data_us)

access_token = 'Bearer BQC1LFGqoWbF2EXfILUkxjsdDVU4yo4yCowUrMqtLm0uMWb4e1I_evZsELADNJSd6UvdJtqvsqQBiu8Ad5s'
url = 'https://api.spotify.com/v1/audio-features/'

track_ids = data_us['ID'].unique()
tracks = []

for _id in track_ids:
    response = requests.get(url + _id, headers={'Authorization': access_token})
    if response.status_code != 200:
        # print(response.text)
        break
    track = response.json()
    tracks.append(track)
    # print(len(tracks))

tracks_df = pd.DataFrame(tracks).drop(['type', 'uri', 'track_href', 'analysis_url'], axis=1)

label_df = data_us[['ID', 'Streams']]
label_df = label_df.groupby('ID').sum().reset_index()
label_df['label'] = np.where(label_df['Streams'] >= label_df['Streams'].mean(), 1, 0)

features = ['acousticness',
            'danceability',
            'duration_ms',
            'energy',
            'instrumentalness',
            'key',
            'liveness',
            'loudness',
            'mode',
            'speechiness',
            'tempo',
            'time_signature',
            'valence']

data = pd.merge(tracks_df, label_df, left_on='id', right_on='ID')
data = data[features + ['label']]
data.to_csv('dataset.csv')

features_df = pd.merge(data_us, tracks_df, left_on='ID', right_on='id')
features_df.to_csv('features_df.csv')
