import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_checks
from sklearn.utils.estimator_checks import check_transformers_unfitted
from sklearn.utils.validation import FLOAT_DTYPES

from sklego.common import flatten
from sklego.preprocessing import ColumnCapper
from tests.conftest import nonmeta_checks

# ColumnCapper works with nan/inf cells
nonmeta_checks_allow_nan_inf = [test for test in nonmeta_checks if test.__name__ != 'check_estimators_nan_inf']


@pytest.mark.parametrize("test_fn", flatten([
    nonmeta_checks_allow_nan_inf,
    # Transformer checks
    check_transformers_unfitted,
    # General checks
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit2d_1feature,
    # ColumnCapper works with 1d arrays. Skipping check_fit1d.
    # estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters,
    estimator_checks.check_transformer_general,
    estimator_checks.check_methods_subset_invariance,
    estimator_checks.check_transformer_data_not_an_array,
    estimator_checks.check_transformer_general,
]))
def test_estimator_checks(test_fn):
    test_fn(ColumnCapper.__name__, ColumnCapper())


def test_quantiles():
    def expect_type_error(min_quantile=0.05, max_quantile=0.95):
        with pytest.raises(TypeError):
            ColumnCapper(min_quantile, max_quantile)

    def expect_value_error(min_quantile=0.05, max_quantile=0.95):
        with pytest.raises(ValueError):
            ColumnCapper(min_quantile, max_quantile)

    # Testing quantiles types
    expect_type_error(min_quantile='a')
    expect_type_error(max_quantile='a')

    # Testing quantiles limits
    expect_value_error(max_quantile=-1)
    expect_value_error(max_quantile=2)
    expect_value_error(min_quantile=-1)
    expect_value_error(min_quantile=2)

    # Testing quantiles order
    expect_value_error(min_quantile=0.5, max_quantile=0.4)


@pytest.fixture()
def df():
    return pd.DataFrame({"a": [1, np.nan, 3, 4],
                         "b": [11, 12, np.inf, 14]})


def test_X_types_and_output_shapes(df):
    capper = ColumnCapper()
    for X in df, df.values, df['a'], df['a'].values:
        # Testing objects with `shape` attribute
        assert capper.fit_transform(X).shape == X.shape
    # Testing a list
    lst = list(df['a'])
    assert capper.fit_transform(lst).shape[0] == len(lst)


def test_nan_inf(df):
    # Capping infs
    capper = ColumnCapper(discard_infs=False)
    assert (capper.fit_transform(df) == np.inf).sum().sum() == 0
    assert np.isnan(capper.fit_transform(df)).sum() == 1

    # Discarding infs
    capper = ColumnCapper(discard_infs=True)
    assert (capper.fit_transform(df) == np.inf).sum().sum() == 0
    assert np.isnan(capper.fit_transform(df)).sum() == 2


def test_pipeline_shape(df):
    pipeline = Pipeline([
        ('cap', ColumnCapper()),
        ('scale', StandardScaler())
    ])

    assert pipeline.fit_transform(df).shape == df.shape


def test_dtype_regression(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert ColumnCapper().fit(X, y).transform(X).dtype in FLOAT_DTYPES


def test_dtype_classification(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert ColumnCapper().fit(X, y).transform(X).dtype in FLOAT_DTYPES
