import numpy as np

from sklego.transformers import RandomAdder


def test_dtype_regression(random_xy_dataset_regr):
    for transformer in [RandomAdder]:
        X, y = random_xy_dataset_regr
        assert transformer().fit(X, y).transform(X).dtype == np.float


def test_dtype_classification(random_xy_dataset_clf):
    for transformer in [RandomAdder]:
        X, y = random_xy_dataset_clf
        assert transformer().fit(X, y).transform(X).dtype == np.float
