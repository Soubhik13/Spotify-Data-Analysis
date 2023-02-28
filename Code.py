import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import datetime
import time

# Reading Data Set 
tracks = pd.read_csv('tracks.csv')
print(tracks)
genre = pd.read_csv('SpotifyFeatures.csv')
print(genre)

tracks.head()
genre.head()

# checking null values
print(pd.isnull(tracks).sum())

print(tracks.info())

print(tracks.describe().transpose())

most = tracks.query('popularity > 90', inplace = False).sort_values('popularity', ascending = False)
most[:10]

least = tracks.sort_values('popularity', ascending = True).head(10)
least

tracks.set_index('release_date', inplace = True)
tracks.index=pd.to_datetime(tracks.index)
print(tracks.head())

print(tracks[['artists']].iloc[18])

tracks['duration'] = tracks['duration_ms'].apply (lambda x : round(x/1000))
tracks.drop('duration_ms', inplace = True, axis=1)
print(tracks.duration.head())

sam = tracks.sample(int(0.004 * len(tracks)))
print(len(sam))

cm = tracks.drop(['key','mode','explicit'], axis=1).corr(method = 'pearson')
plt.figure(figsize=(14,6))
map = sns.heatmap(cm, annot = True, fmt = '.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
map.set_title('Correlation Heatmap between Variable')
map.set_xticklabels(map.get_xticklabels(), rotation=90)

plt.figure(figsize=(10,6))
sns.regplot(data=sam, y='loudness', x='energy', color='c').set(title='Loudness vs Energy Correlation')

plt.figure(figsize=(10,6))
sns.regplot(data=sam, y='popularity', x='acousticness', color='b').set(title='Popularity vs Acousticness Correlation')

tracks['dates']=tracks.index.get_level_values('release_date')
tracks.dates=pd.to_datetime(tracks.dates)
years=tracks.dates.dt.year
print(tracks.head())

sns.displot(years, discrete=True, aspect=2, height=5, kind='hist').set(title='Number of songs per year')

total_dr = tracks.duration
fig_dims = (18,7)
fig, ax = plt.subplots(figsize=fig_dims)
fig = sns.barplot(x = years, y = total_dr, ax = ax, errwidth = False).set(title='Years vs Duration')
plt.xticks(rotation=90)

plt.title('Duration of songs in different Genres')
sns.color_palette('rocket', as_cmap=True)
sns.barplot(y='genre', x='duration_ms', data=genre)
plt.xlabel('Duration in milliseconds')
plt.ylabel('Genres')

sns.set_style(style='darkgrid')
plt.figure(figsize=(10,5))
popular = genre.sort_values('popularity', ascending=False).head(10)
sns.barplot(y = 'genre', x = 'popularity', data = popular).set(title='Top 5 Genres by Popularity')

print("Tensorflow is running",end=" ")
for i in range(5):
    print(".", end="")
    time.sleep(1)
print()
mnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(120,activation=tf.nn.relu),tf.keras.layers.Dense(20,activation=tf.nn.softmax)])
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=18)
#model.evaluate(test_images,test_labels)