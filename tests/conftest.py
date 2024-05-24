import itertools as it

import numpy as np
import pandas as pd
import polars as pl
import pytest

n_vals = (10, 500)
k_vals = (1, 5)
np_types = (np.int32, np.float32, np.float64)


def select_tests(include, exclude=[]):
    """Return an iterable of include with all tests whose name is not in exclude"""
    for test in include:
        if test.__name__ not in exclude:
            yield test


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_regr(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,))
    return X, y


@pytest.fixture(scope="module", params=[_ for _ in it.product([10, 100], [1, 2, 3], np_types)])
def random_xy_dataset_regr_small(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,))
    return X, y


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_clf(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,)) > 0.0
    return X, y


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_multiclf(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = pd.cut(np.random.normal(0, 2, (n,)), 3).codes
    return X, y


@pytest.fixture(scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)])
def random_xy_dataset_multitarget(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.randint(0, 2, (n, k)) > 0.0
    return X, y


@pytest.fixture
def sensitive_classification_dataset():
    df = pd.DataFrame(
        {
            "x1": [1, 0, 1, 0, 1, 0, 1, 1],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1],
            "y": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )

    return df[["x1", "x2"]], df["y"]


@pytest.fixture(params=[pd.DataFrame, pl.DataFrame])
def funct(request):
    return request.param


@pytest.fixture
def sensitive_classification_dataset_equalopportunity(funct):
    df = funct(
        {
            "x1": [1, 0, 1, 0, 1, 0, 1, 1],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1],
            "y": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )

    return df[["x1", "x2"]], df["y"]


@pytest.fixture
def sensitive_multiclass_classification_dataset():
    df = pd.DataFrame(
        {
            "x1": [1, 0, 1, 0, 1, 0, 1, 1, -2, -2, -2, -2],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            "y": [1, 1, 1, 0, 1, 0, 0, 0, 2, 2, 0, 0],
        }
    )
    return df[["x1", "x2"]], df["y"]


def id_func(param):
    """Returns the repr of an object for usage in pytest parametrize"""
    return repr(param)
