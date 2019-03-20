import pytest
import pandas as pd
import itertools as it
import numpy as np
from sklego.preprocessing import PandasTypeSelector
from tests.conftest import id_func


@pytest.mark.parametrize("transformer", [
    PandasTypeSelector(include=['number']),
], ids=id_func)
def test_len_regression(transformer, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    X = pd.DataFrame(X)
    assert transformer.fit(X, y).transform(X).shape[0] == X.shape[0]


@pytest.mark.parametrize("transformer", [
    PandasTypeSelector(include=['number']),
], ids=id_func)
def test_len_classification(transformer, random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X = pd.DataFrame(X)
    assert transformer.fit(X, y).transform(X).shape[0] == X.shape[0]


@pytest.mark.parametrize('include,exclude', [_ for _ in it.combinations(['number', 'datetime', 'timedelta',
                                                                         'category', 'datetimetz', ''], 2)])
def test_get_params_str(include, exclude):
    transformer = PandasTypeSelector(include=include, exclude=exclude)

    assert transformer.get_params() == {
        'include': include,
        'exclude': exclude
    }


@pytest.mark.parametrize('include,exclude', [_ for _ in it.combinations([np.int64, np.float64, np.datetime64,
                                                                         np.timedelta64], 2)])
def test_get_params_np(include, exclude):
    transformer = PandasTypeSelector(include=include, exclude=exclude)

    assert transformer.get_params() == {
        'include': include,
        'exclude': exclude
    }