import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class ColumnsCapper(TransformerMixin):
    
    def __init__(self, quantile_thresholds, cols_names=None, absolute_thresholds=None, use_quantile=True):
        self.cols_names = cols_names
        self.absolute_thresholds = absolute_thresholds
        self.quantile_thresholds = quantile_thresholds
        self.use_quantile = use_quantile
        
    def transform(self, X):
        if self.cols_names is None:
            self.cols_names = X.columns
    
        if self.use_quantile:

            if type(self.quantile_thresholds) is float:
                for col in self.cols_names:
                    q = X[col].quantile(self.quantile_thresholds, interpolation='linear')
                    X[col] = X[col].apply(lambda x: x if (x < q) else q)

            if type(self.quantile_thresholds) is list:
                for col, q_t in zip(self.cols_names, self.quantile_thresholds):
                    q = X[col].quantile(q=q_t, axis=1, interpolation='linear')
                    X[col] = X[col].apply(lambda x: x if (x < q) else q)
        else:

            if type(self.absolute_thresholds) is float:
                for col in self.cols_names:
                    X[col] = X[col].apply(lambda x: x if (x < self.absolute_thresholds) else self.absolute_thresholds)

            if type(self.absolute_thresholds) is list:
                for col, threshold in zip(self.cols_names, self.thresholds):
                    X[col] = X[col].apply(lambda x: x if (x < self.threshold) else self.threshold)
        return X
    
    def fit(self):
        return self