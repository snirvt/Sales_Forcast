import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def predict_with_lm(model,lm, X, years):
    linear_pred = lm.predict(years)
    preds = model.predict(X)
    return np.clip(preds - linear_pred, 0, None)

def predict(model, X):
    preds = model.predict(X)
    preds = np.clip(preds, 0, None)
    return preds


class UnnamedDrop(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(['Unnamed: 0'], axis=1)


class ExtractDates(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['date'] = pd.to_datetime(X['date'])
        X['dow'] = X['date'].dt.isocalendar().day
        X['woy'] = X['date'].dt.isocalendar().week
        X['year'] = X['date'].dt.isocalendar().year
        X = X.drop(['date'], axis=1)
        return X


class OHE(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=['store_id', 'sku_category'])
        return X


class OHE_stores(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=['store_id', 'sku_category', 'type'])
        return X
