from tqdm import tqdm
import polars as pl
import numpy as np

music = pl.read_csv('data/charts.csv')
data = pl.read_csv('data/prices.csv')
music = music.filter(pl.col('chart') == 'top200')
music = music.drop(['artist', 'chart', 'rank', 'url'])

music = music.filter(pl.col('title').count().over('title') >= 3000)
# music = music.with_columns(pl.col('title').cast(pl.Categorical).to_physical())

music = music.with_columns(pl.col('streams').sum().over('date').alias('S'))

print(music)
print(music['title'].max())
print(data)
print(music['title'].value_counts().sort('count'))
# print(music.filter(pl.col('title') == 'Shape of You'))

dates = data['Date']
Y = np.concat([np.diff(data['XLU_Close'].to_numpy()), [0]])
print(Y.shape, len(dates))

X = []

for d in tqdm(dates):
    v = music.filter(pl.col('date') == d)['S'].to_list()
    if len(v) > 0:
        X.append(v[0])
    else:
        X.append(0)
X = np.array(X)
print(X.shape)
