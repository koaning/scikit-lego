import itertools as it

import numpy as np

import pytest

n_vals = (1, 100, 1000, 10000)
k_vals = (1, 5, 10, 25)
np_types = (np.int32, np.float32, np.float64)


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_regr(request):
    n, k, np_type = request.param
    X = np.random.normal(0, 1, (n, k)).astype(np_type)
    y = np.random.normal(0, 1, (n, 1))
    return X, y


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_clf(request):
    n, k, np_type = request.param
    X = np.random.normal(0, 1, (n, k)).astype(np_type)
    y = np.random.normal(0, 1, (n, 1)) > 0.0
    return X, y
