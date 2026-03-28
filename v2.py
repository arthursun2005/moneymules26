from tqdm import tqdm
import polars as pl
import numpy as np

K = 100

music = pl.read_csv('data/charts.csv')
data = pl.read_csv('data/prices.csv')
music = music.filter(pl.col('chart') == 'top200')
music = music.drop(['artist', 'chart', 'rank', 'url'])

T = music.group_by('title').agg(pl.col('streams').sum().alias('total')).sort('total', descending=True).head(K)
# print(T)

titles = T['title']

music = music.filter(pl.col('title').is_in(titles))
# print(music)

music = music.with_columns(pl.col('streams').sum().over('title', 'date').alias('S'))
music = music.unique(['title', 'date'])
print(music)


# print(music)
# print(music['title'].max())
# print(data)
# print(music['title'].value_counts().sort('count'))
# print(music.filter(pl.col('title') == 'Shape of You'))

dates = data['Date']
Y = np.concat([np.diff(data['XLU_Close'].to_numpy()), [0]])
Y = np.concat([np.diff(data['XLP_Close'].to_numpy()), [0]])
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

print(np.corrcoef(X, Y))
