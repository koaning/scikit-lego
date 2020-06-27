import numpy as np
import pandas as pd
import pytest
from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.preprocessing import DictMapper
from tests.conftest import general_checks, transformer_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            transformer_checks,
            general_checks,
            # nonmeta_checks,
            estimator_checks.check_estimators_dtypes,
            estimator_checks.check_fit_score_takes_y,
            # Dtype can be anything
            # estimator_checks.check_dtype_object,
            estimator_checks.check_sample_weights_pandas_series,
            estimator_checks.check_sample_weights_list,
            estimator_checks.check_sample_weights_invariance,
            estimator_checks.check_estimators_fit_returns_self,
            estimator_checks.check_complex_data,
            estimator_checks.check_estimators_empty_data_messages,
            estimator_checks.check_pipeline_consistency,
            estimator_checks.check_estimators_nan_inf,
            estimator_checks.check_estimators_overwrite_params,
            estimator_checks.check_estimator_sparse_data,
            estimator_checks.check_estimators_pickle,
        ]
    ),
)
def test_estimator_checks(test_fn):
    test_fn(DictMapper.__name__, DictMapper(mapper={"foo": 1}, default=-1))


@pytest.fixture()
def mapper():
    return {"foo": 1, "bar": 2, "baz": 3}


@pytest.mark.parametrize(
    "input_array,expected_array",
    [
        (["foo", "bar", "baz"], [1, 2, 3]),
        (["foo", "bar", "bar"], [1, 2, 2]),
        (["foo", "bar", "monty"], [1, 2, -1]),
        (["foo", "bar", np.nan], [1, 2, -1]),
        ([["foo", "bar", "baz"], ["foo", "bar", "baz"]], [[1, 2, 3], [1, 2, 3]]),
    ],
)
def test_array(input_array, expected_array, mapper):
    X = np.array(input_array).reshape(-1, 1)
    expected = np.array(expected_array).reshape(-1, 1)
    result = DictMapper(mapper=mapper, default=-1).fit_transform(X)
    np.testing.assert_array_equal(result, expected)


def test_pandas(mapper):
    X = pd.DataFrame(["foo", "bar", "baz"], dtype=object)
    expected = np.array([1, 2, 3]).reshape(-1, 1)
    result = DictMapper(mapper=mapper, default=-1).fit_transform(X)
    np.testing.assert_array_equal(result, expected)


def test_no_mapper():
    mapper = {}
    X = pd.DataFrame(["foo", "bar", "baz"], dtype=object)
    expected = np.array([-1, -1, -1]).reshape(-1, 1)
    result = DictMapper(mapper=mapper, default=-1).fit_transform(X)
    np.testing.assert_array_equal(result, expected)
