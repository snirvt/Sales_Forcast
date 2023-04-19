import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from utils import predict_with_lm, UnnamedDrop, ExtractDates, OHE
from sklearn.linear_model import LinearRegression


### Loading the data
train = pd.read_csv('data/train.csv', index_col=0)
stores = pd.read_csv('data/stores.csv', index_col=0)
is_weighted = False
use_stores = False
### Seperating X and y
train_X, train_y = train.loc[:, train.columns != 'sales'].copy(), train['sales'].copy()
if use_stores:
    train_X = train_X.join(stores.set_index('store_id'), on='store_id')

### Creating pipline
pipe = Pipeline([('drop', UnnamedDrop()),
                 ('dates', ExtractDates()),
                 ("encoder", OHE()),
                 ])
train_X = pipe.fit_transform(train_X)
train_X, train_year = train_X.loc[:, train_X.columns != 'year'], train_X[['year','tot_promoted']]

val_size = np.count_nonzero(train_year==2017)
weights = np.ones(len(train_year[:-val_size]))
if is_weighted:
    year2weight = {2013: 1, 2014: 1.1, 2015: 1.2, 2016: 1.3, 2017: 1.4}
    weights = [year2weight[num] for num in train_year[:-val_size]['year']]
cat_features = []
if use_stores:
    cat_features = ['type', 'province', 'geo', 'group']
depth = [5, 10, 15]
lrs = [0.05, 0.1, 0.2]
best_r2 = 0
best_key = (depth[0], lrs[0])
results = {}
for d in depth:
    for lr in lrs:
        lr_model = LinearRegression()
        lr_model.fit(train_year.iloc[:-val_size].values.reshape(-1, 2), train_y.iloc[:-val_size])
        linear_pred_past = lr_model.predict(train_year.iloc[:-val_size].values.reshape(-1, 2))
        linear_pred_future = lr_model.predict(train_year.iloc[-val_size:].values.reshape(-1, 2))

        model = CatBoostRegressor(iterations=150, depth=d, learning_rate=lr, loss_function='RMSE',
                                  has_time=True, cat_features=cat_features)
        model.fit(train_X.iloc[:-val_size, :], linear_pred_past + train_y.iloc[:-val_size],
                  eval_set=(train_X.iloc[-val_size:, :], linear_pred_future + train_y.iloc[-val_size:])
                  )

        model_con = CatBoostRegressor(iterations=100, depth=d, learning_rate=lr/2, loss_function='RMSE',
                                      has_time=True, cat_features=cat_features)
        model_con.fit(train_X.iloc[:-val_size, :], linear_pred_past + train_y.iloc[:-val_size],
                      eval_set=(train_X.iloc[-val_size:, :], linear_pred_future + train_y.iloc[-val_size:], )
                      , init_model=model, sample_weight=weights)
        model = model_con

        pred = predict_with_lm(model, lr_model, train_X.iloc[-val_size:, :],
                               train_year.iloc[-val_size:].values.reshape(-1, 2))
        r2 = r2_score(train_y.iloc[-val_size:], pred)
        print(f'r^2 of {r2} on validation set')
        results[d, lr] = (r2, pred)
        if r2>best_r2:
            best_r2 = r2
            best_key = (d,lr)




title = 'Model Comparison'
if is_weighted:
    title +=' weighted'
if use_stores:
    title += ' stores'

keys_list = list(results.keys())
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 14))
for i, axi in enumerate(ax.flat):
    d, lr = keys_list[i]
    r2 = results[keys_list[i]][0]
    pred = results[keys_list[i]][1]
    axi.scatter(train_y.iloc[-val_size:], pred)
    axi.set_xlabel('True Sales')
    if i % 3 == 0:
        axi.set_ylabel('Predicted Sales')
    axi.set_title(f'Depth: {d} lr: {lr} r^2: {np.round(r2, 3)}')
fig.suptitle(title, fontsize=25)
plt.savefig(f'{title}.png'.replace(' ', '_'))
plt.close()


date_to_subtract = pd.to_datetime('2016-12-31')
date_diff = pd.to_datetime(train['date'].iloc[-val_size:]) - date_to_subtract
err = np.abs((train_y.iloc[-val_size:] - results[best_key][1]))
err = err.values
days = date_diff.dt.days.values
days_set = np.sort(list(set(days)))
means = []
stds = []
for d_ in days_set:
    idx = np.argwhere(d_== days)
    means.append(err[idx].mean())
    stds.append(err[idx].std())
plt.errorbar(days_set[::2], means[::2], yerr=stds[::2], fmt='o', capsize=2)
plt.title('Error vs Time')
plt.xlabel('Days After 2016-12-31')
plt.ylabel('Error')
plt.savefig(f'{title}_err_time.png'.replace(' ', '_'))
plt.close()