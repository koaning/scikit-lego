import numpy as np


class WontPredictError(Exception):
    pass


def _handle_behavior(pred, wont_pred, behavior):
    pred = pred.copy()
    if isinstance(behavior, str):
        if behavior not in ['nan', 'error']:
            raise ValueError(f"Got {behavior} while it can only be either 'nan' or 'error'.")
    if behavior == 'nan':
        pred[wont_pred] = np.nan
        return pred
    if behavior == 'error':
        if wont_pred.sum() > 0:
            raise WontPredictError("We should not predict here.")
        return pred
    pred[wont_pred] = behavior
    return pred


class WontPredictThreshold:
    def __init__(self, estimator, threshold=0.2, behavior='nan'):
        self.estimator = estimator
        self.threshold = threshold
        self.behavior = behavior

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        proba = self.estimator.predict_proba(X)
        proba_sort = np.sort(proba, axis=1)
        diff = proba_sort[:, -1] - proba_sort[:, -2]
        return _handle_behavior(pred=self.estimator.predict(X), wont_pred=diff > self.threshold, behavior=self.behavior)

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)
        proba_sort = np.sort(proba, axis=1)
        diff = proba_sort[:, -1] - proba_sort[:, -2]
        return _handle_behavior(pred=proba, wont_pred=diff > self.threshold, behavior=self.behavior)


class WontPredictOutlier:
    def __init__(self, estimator, outlier_estimator, behavior='nan'):
        self.estimator = estimator
        self.outlier_estimator = outlier_estimator
        self.behavior = behavior

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.outlier_estimator.fit(X)
        return self

    def predict(self, X):
        outlier = self.outlier_estimator.predict(X)
        return _handle_behavior(pred=self.estimator.predict(X), wont_pred=outlier, behavior=self.behavior)

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)
        outlier = self.outlier_estimator.predict(X)
        return _handle_behavior(pred=proba, wont_pred=outlier, behavior=self.behavior)
