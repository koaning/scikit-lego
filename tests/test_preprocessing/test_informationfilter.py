import pytest
import numpy as np
import pandas as pd

from sklearn.utils import estimator_checks
from sklearn.datasets import load_boston

from sklego.common import flatten
from sklego.preprocessing import InformationFilter
from tests.conftest import transformer_checks, general_checks


@pytest.mark.parametrize("test_fn", flatten([
    transformer_checks,
]))
def test_estimator_checks(test_fn):
    test_fn(InformationFilter.__name__, InformationFilter(columns=[0]))


def test_v_columns_orthogonal():
    X, y = load_boston(return_X_y=True)
    ifilter = InformationFilter(columns=[11, 12]).fit(X, y)
    v_values = ifilter._make_v_vectors(X, [11, 12])
    assert v_values.prod(axis=1).sum() == pytest.approx(0, abs=1E-5)


def test_output_orthogonal():
    X, y = load_boston(return_X_y=True)
    X_fair = InformationFilter(columns=[11, 12]).fit_transform(X)
    assert all([(c * X[:, 11]).sum() < 1E-5 for c in X_fair.T])
    assert all([(c * X[:, 12]).sum() < 1E-5 for c in X_fair.T])


def test_output_orthogonal_pandas():
    X, y = load_boston(return_X_y=True)
    df = pd.DataFrame(X, columns=['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                                  'age', 'dis', 'rad', 'tax', 'ptratio', 'b',
                                  'lstat'])
    X_fair = InformationFilter(columns=["b", "lstat"]).fit_transform(df)
    assert all([(c * df["b"]).sum() < 1E-5 for c in X_fair.T])
    assert all([(c * df["lstat"]).sum() < 1E-5 for c in X_fair.T])


def test_output_orthogonal_general_cols():
    X, y = load_boston(return_X_y=True)
    cols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
            'age', 'dis', 'rad', 'tax', 'ptratio', 'b',
            'lstat']
    df = pd.DataFrame(X, columns=cols)
    for col in cols:
        X_fair = InformationFilter(columns=col).fit_transform(df)
        assert all([(c * df[col]).sum() < 1E-5 for c in X_fair.T])