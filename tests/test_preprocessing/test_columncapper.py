import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_checks
from sklearn.utils.estimator_checks import check_transformers_unfitted

from sklego.common import flatten
from sklego.preprocessing import ColumnCapper

@pytest.mark.parametrize("test_fn", flatten([
    # Transformer checks
    check_transformers_unfitted,
    # General checks
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters
]))

def test_estimator_checks(test_fn):
    test_fn(ColumnCapper.__name__, ColumnCapper())

@pytest.fixture()
def df():
    return pd.DataFrame({"a": [1, 2, 3, 4],
                         "b": [11, 12, np.inf, 14]})

def test_X_types_and_output_shapes(df):
    capper = ColumnCapper()
    for X in df, df.values, df['a'], df['a'].values:
        assert capper.fit_transform(X).shape == X.shape

def test_infs(df):
    capper = ColumnCapper(discard_infs=False)
    assert (capper.fit_transform(df) == np.inf).sum().sum() == 0
    assert (capper.fit_transform(df) == np.nan).sum().sum() == 0

    capper = ColumnCapper(discard_infs=True)
    assert (capper.fit_transform(df) == np.inf).sum().sum() == 0
    assert np.isnan(capper.fit_transform(df)).sum() == 1

def test_pipeline_shape(df):
    pipeline = Pipeline([
        ('cap', ColumnCapper()),
        ('scale', StandardScaler())
    ])

    assert pipeline.fit_transform(df).shape == df.shape
