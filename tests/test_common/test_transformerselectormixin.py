import itertools as it

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklego.common import TrainOnlyTransformerMixin
from tests.conftest import np_types, n_vals, k_vals


class TrainOnlyTrainOnlyTransformer(TrainOnlyTransformerMixin, BaseEstimator):

    def fit(self, X, y):
        super().fit(X, y)

    def transform_train(self, X, y=None):
        return X + np.random.normal(0, 1, size=X.shape)


def test_hash_numpy():
    """Tests whether the hash function does not produce collisions on np arrays"""
    hashes = []
    for n, k, np_type in it.product(n_vals, k_vals, np_types):
        X = np.random.normal(0, 2, (n, k)).astype(np_type)

        hashes.append(TrainOnlyTransformerMixin._hash(X))

    assert len(hashes) == len(set(hashes))


def test_hash_pandas():
    """Tests whether the hash function does not produce collisions on dataframes"""
    hashes = []
    for n, k, np_type in it.product(n_vals, k_vals, np_types):
        X = pd.DataFrame(np.random.normal(0, 2, (n, k)).astype(np_type))

        hashes.append(TrainOnlyTransformerMixin._hash(X))

    assert len(hashes) == len(set(hashes))


def test_bare_trainonlytransformer(random_xy_dataset_regr):
    """Tests whether the trainonlytransformer will only transform train when used directly"""

    X_train, X_test, y_train, y_test = train_test_split(*random_xy_dataset_regr)

    trf = TrainOnlyTrainOnlyTransformer()
    trf.fit(X_train, y_train)

    assert np.all(trf.transform(X_train) != X_train)
    assert np.all(trf.transform(X_test) == X_test)


def test_pipeline_trainonlytransformer(random_xy_dataset_regr):
    """Tests whether the trainonlytransformer will only transform train when used in a pipeline"""

    X_train, X_test, y_train, y_test = train_test_split(*random_xy_dataset_regr)

    trf = make_pipeline(TrainOnlyTrainOnlyTransformer())
    trf.fit(X_train, y_train)

    assert np.all(trf.transform(X_train) != X_train)
    assert np.all(trf.transform(X_test) == X_test)


def test_bare_trainonlytransformer_pandas(random_xy_dataset_regr):
    """Tests whether the trainonlytransformer will only transform train when used directly"""
    X, y = pd.DataFrame(random_xy_dataset_regr[0]), pd.DataFrame(random_xy_dataset_regr[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    trf = TrainOnlyTrainOnlyTransformer()
    trf.fit(X_train, y_train)

    assert np.all(trf.transform(X_train) != X_train)
    assert np.all(trf.transform(X_test) == X_test)

    assert isinstance(trf.transform(X_train), pd.DataFrame)
    assert isinstance(trf.transform(X_test), pd.DataFrame)


def test_pipeline_trainonlytransformer_pandas(random_xy_dataset_regr):
    """Tests whether the trainonlytransformer will only transform train when used in a pipeline"""
    X, y = pd.DataFrame(random_xy_dataset_regr[0]), pd.DataFrame(random_xy_dataset_regr[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    trf = make_pipeline(TrainOnlyTrainOnlyTransformer())
    trf.fit(X_train, y_train)

    assert np.all(trf.transform(X_train) != X_train)
    assert np.all(trf.transform(X_test) == X_test)

    assert isinstance(trf.transform(X_train), pd.DataFrame)
    assert isinstance(trf.transform(X_test), pd.DataFrame)
