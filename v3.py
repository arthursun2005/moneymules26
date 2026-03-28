from tqdm import tqdm
import polars as pl
import numpy as np
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression

K = 64

music = pl.read_csv('data/charts.csv')
data = pl.read_csv('data/prices.csv')
music = music.filter(pl.col('chart') == 'top200')
music = music.drop(['artist', 'chart', 'rank', 'url'])

T = music.group_by('title').agg(pl.col('streams').sum().alias('total')).sort('total', descending=True).head(K)

titles = T['title']

music = music.filter(pl.col('title').is_in(titles))

music = music.with_columns(pl.col('streams').sum().over('title', 'date').alias('S'))
music = music.unique(['title', 'date'])
print(music)

dates = data['Date']
Y = np.concat([np.diff(data['XLU_Close'].to_numpy()), [0]])
# Y = np.concat([np.diff(data['XLP_Close'].to_numpy()), [0]])
print(Y.shape, len(dates))

X = []

for d in tqdm(dates):
    s = []
    for t in titles:
        v = music.filter((pl.col('title') == t) & (pl.col('date') == d))['S'].to_list()
        if len(v) > 0:
            s.append(v[0])
        else:
            s.append(0)
    X.append(s)
X = np.array(X)
print(X.shape)
print(X, Y)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y)

# model = cb.CatBoostRegressor()
# model = xgb.XGBRegressor(
model = LinearRegression()
model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))
print(model.score(X_val, Y_val))
# print(np.corrcoef(X, Y))
