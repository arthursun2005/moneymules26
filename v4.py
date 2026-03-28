from tqdm import tqdm
import polars as pl
import numpy as np
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression

music = pl.read_csv('data/charts.csv')
data = pl.read_csv('data/prices.csv')
music = music.filter(pl.col('chart') == 'top200')
music = music.drop(['artist', 'chart', 'rank', 'url'])

regions = music['region'].unique()

music = music.with_columns(pl.col('streams').sum().over('region', 'date').alias('S'))
music = music.unique(['region', 'date'])
print(music)

dates = data['Date']
Y = np.concat([np.diff(data['XLU_Close'].to_numpy()), [0]])
# Y = np.concat([np.diff(data['XLP_Close'].to_numpy()), [0]])
print(Y.shape, len(dates))

X = []

for d in tqdm(dates):
    s = []
    for r in regions:
        v = music.filter((pl.col('region') == r) & (pl.col('date') == d))['S'].to_list()
        if len(v) > 0:
            s.append(v[0])
        else:
            s.append(0)
    X.append(s)
X = np.array(X)
print(X.shape)
print(X, Y)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y)

model = cb.CatBoostRegressor()
# model = xgb.XGBRegressor(
# model = LinearRegression()
model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))
print(model.score(X_val, Y_val))
# print(np.corrcoef(X, Y))
