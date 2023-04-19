import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from utils import predict, predict_with_lm, UnnamedDrop, ExtractDates, OHE
from sklearn.linear_model import LinearRegression

def run():
    ### Loading the data
    train = pd.read_csv('data/train.csv', index_col=0)
    test = pd.read_csv('data/test.csv', index_col=0)
    stores = pd.read_csv('data/stores.csv', index_col=0)

    ### Seperating X and y
    train_X, train_y = train.loc[:, train.columns != 'sales'], train['sales']
    train_X = train_X.join(stores.set_index('store_id'), on='store_id')
    test = test.join(stores.set_index('store_id'), on='store_id')

    ### Creating pipline
    pipe = Pipeline([('drop', UnnamedDrop()),
                     ('dates', ExtractDates()),
                     ("encoder", OHE()),
                     ])

    train_X = pipe.fit_transform(train_X)
    test = pipe.transform(test)

    train_X, train_year = train_X.loc[:, train_X.columns != 'year'], train_X[['year', 'tot_promoted']]
    test, test_year = test.loc[:, test.columns != 'year'], test[['year', 'tot_promoted']]

    ### Initialising model
    val_size = 1000
    # val_size = np.count_nonzero(train_year==2017)

    lr_model = LinearRegression()
    lr_model.fit(train_year.iloc[:-val_size].values.reshape(-1,2), train_y.iloc[:-val_size])
    linear_pred_past = lr_model.predict(train_year.iloc[:-val_size].values.reshape(-1, 2))
    linear_pred_future = lr_model.predict(train_year.iloc[-val_size:].values.reshape(-1, 2))

    model = CatBoostRegressor(iterations=150, depth=15, learning_rate=0.1, loss_function='RMSE',
                              has_time=True, cat_features = ['type', 'province', 'geo', 'group'])
    model.fit(train_X.iloc[:-val_size, :], linear_pred_past + train_y.iloc[:-val_size],
              eval_set=(train_X.iloc[-val_size:, :], linear_pred_future + train_y.iloc[-val_size:])
              )

    ### Finetuning
    for lr in [0.05]:
        model_con = CatBoostRegressor(iterations=100, depth=15, learning_rate=lr, loss_function='RMSE',
                                      has_time=True, cat_features = ['type', 'province', 'geo', 'group'])
        model_con.fit(train_X.iloc[:-val_size, :], linear_pred_past + train_y.iloc[:-val_size],
                      eval_set=(train_X.iloc[-val_size:, :], linear_pred_future + train_y.iloc[-val_size:])
                      , init_model=model)
        model = model_con

    ### Generate validation plots
    pred = predict_with_lm(model, lr_model, train_X.iloc[-val_size:, :], train_year.iloc[-val_size:].values.reshape(-1,2))

    r2 = r2_score(train_y.iloc[-val_size:], pred)
    print(f'r^2 of {r2} on validation set')
    plt.scatter(train_y.iloc[-val_size:], pred)
    m, b = np.polyfit(train_y.iloc[-val_size:], pred, deg=1)
    plt.axline(xy1=(0, b), slope=m, color='r', label=f'$y = {m:.3f}x {b:+.2f}$')
    plt.title(f'Validation Set Goodness of Fit : r^2 = {np.round(r2, 3)}')
    plt.xlabel('True Sales')
    plt.ylabel('Predicted Sales')
    plt.legend()
    plt.savefig('validation_results.png')
    plt.close()
    print('Validation results plot saved in validation_results.png')

    ### Save test predictions into csv
    pred_test = predict_with_lm(model, lr_model, test.iloc[-val_size:, :], test_year.iloc[-val_size:].values.reshape(-1, 2))
    pred_test_pd = pd.DataFrame(pred_test, columns=['Predicted_Sales'])
    pred_test_pd.to_csv('Predicted_Sales.csv')
    print('Test predicions saved in Predicted_Sales.csv')


if __name__ == '__main__':
    run()
