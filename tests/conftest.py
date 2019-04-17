import itertools as it

import numpy as np

import pytest

n_vals = (10, 100, 10000)
k_vals = (1, 2, 25)
np_types = (np.int32, np.float32, np.float64)


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_regr(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n, ))
    return X, y


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_clf(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n, )) > 0.0
    return X, y


def id_func(param):
    """Returns the repr of an object for usage in pytest parametrize"""
    return repr(param)
