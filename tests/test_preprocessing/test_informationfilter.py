import pytest
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import estimator_checks
from sklearn.datasets import load_boston

from sklego.common import flatten
from sklego.preprocessing import InformationFilter
from tests.conftest import transformer_checks, general_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            transformer_checks,
            general_checks,
            # nonmeta_checks
            estimator_checks.check_estimators_dtypes,
            estimator_checks.check_fit_score_takes_y,
            estimator_checks.check_dtype_object,
            estimator_checks.check_sample_weights_pandas_series,
            estimator_checks.check_sample_weights_list,
            estimator_checks.check_sample_weights_invariance,
            estimator_checks.check_estimators_fit_returns_self,
            estimator_checks.check_complex_data,
            # this won't work because we need to select a column
            # estimator_checks.check_estimators_empty_data_messages,
            estimator_checks.check_pipeline_consistency,
            estimator_checks.check_estimators_nan_inf,
            estimator_checks.check_estimators_overwrite_params,
            estimator_checks.check_estimator_sparse_data,
            estimator_checks.check_estimators_pickle,
        ]
    ),
)
def test_estimator_checks(test_fn):
    test_fn(InformationFilter.__name__, InformationFilter(columns=[0]))


def test_v_columns_orthogonal():
    X, y = load_boston(return_X_y=True)
    ifilter = InformationFilter(columns=[11, 12]).fit(X, y)
    v_values = ifilter._make_v_vectors(X, [11, 12])
    assert v_values.prod(axis=1).sum() == pytest.approx(0, abs=1e-5)


def test_output_orthogonal():
    X, y = load_boston(return_X_y=True)
    X_fair = InformationFilter(columns=[11, 12]).fit_transform(X)
    assert all([(c * X[:, 11]).sum() < 1e-5 for c in X_fair.T])
    assert all([(c * X[:, 12]).sum() < 1e-5 for c in X_fair.T])


def test_alpha_param1():
    X, y = load_boston(return_X_y=True)
    ifilter = InformationFilter(columns=[11, 12], alpha=0.0)
    X_removed = np.delete(X, [11, 12], axis=1)
    assert np.isclose(ifilter.fit_transform(X), X_removed).all()


def test_alpha_param2():
    X, y = load_boston(return_X_y=True)
    df = pd.DataFrame(
        X,
        columns=[
            "crim",
            "zn",
            "indus",
            "chas",
            "nox",
            "rm",
            "age",
            "dis",
            "rad",
            "tax",
            "ptratio",
            "b",
            "lstat",
        ],
    )
    ifilter = InformationFilter(columns=["b", "lstat"], alpha=0.0)
    X_removed = df.drop(columns=["b", "lstat"]).values
    assert np.isclose(ifilter.fit_transform(df), X_removed).all()


def test_output_orthogonal_pandas():
    X, y = load_boston(return_X_y=True)
    df = pd.DataFrame(
        X,
        columns=[
            "crim",
            "zn",
            "indus",
            "chas",
            "nox",
            "rm",
            "age",
            "dis",
            "rad",
            "tax",
            "ptratio",
            "b",
            "lstat",
        ],
    )
    X_fair = InformationFilter(columns=["b", "lstat"]).fit_transform(df)
    assert all([(c * df["b"]).sum() < 1e-5 for c in X_fair.T])
    assert all([(c * df["lstat"]).sum() < 1e-5 for c in X_fair.T])


def test_output_orthogonal_general_cols():
    X, y = load_boston(return_X_y=True)
    cols = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ]
    df = pd.DataFrame(X, columns=cols)
    for col in cols:
        X_fair = InformationFilter(columns=col).fit_transform(df)
        assert all([(c * df[col]).sum() < 1e-5 for c in X_fair.T])


def test_pipeline_gridsearch():
    X, y = load_boston(return_X_y=True)
    pipe = Pipeline(
        [("info", InformationFilter(columns=[11, 12])), ("model", LinearRegression())]
    )
    mod = GridSearchCV(
        estimator=pipe, param_grid={"info__columns": [[], [11], [12], [11, 12]]}, cv=2
    )
    assert pd.DataFrame(mod.fit(X, y).cv_results_).shape[0] == 4
