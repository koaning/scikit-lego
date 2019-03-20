import pytest
import pandas as pd
import itertools as it
import numpy as np
from sklego.preprocessing import PandasTypeSelector


@pytest.mark.parametrize('include,exclude', [_ for _ in it.combinations(['number', 'datetime', 'timedelta',
                                                                         'category', 'datetimetz', None], 2)])
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


def test_value_error_empty(random_xy_dataset_regr):
    transformer = PandasTypeSelector(exclude=['number'])
    X, y = random_xy_dataset_regr
    X = pd.DataFrame(X)

    with pytest.raises(ValueError):
        transformer.fit(X, y)


def test_value_error_inequal(random_xy_dataset_regr):
    transformer = PandasTypeSelector(include=['number'])
    X, y = random_xy_dataset_regr
    X = pd.DataFrame(X)
    if X.shape[0] > 0:
        with pytest.raises(ValueError):
            transformer.fit(X)
            # Remove column to create error
            transformer.transform(X.iloc[:, :-1])
