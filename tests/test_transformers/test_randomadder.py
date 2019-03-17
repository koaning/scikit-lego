import numpy as np

from sklego.transformers import RandomAdder


def test_dtype_regression(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert RandomAdder().fit(X, y).transform(X).dtype == np.float


def test_dtype_classification(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert RandomAdder().fit(X, y).transform(X).dtype == np.float
