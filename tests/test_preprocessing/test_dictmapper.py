import numpy as np
import pandas as pd
import pytest

from sklego.common import flatten
from sklego.preprocessing import DictMapper


from tests.conftest import select_tests, transformer_checks, general_checks, nonmeta_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, transformer_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_dtype_object",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
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
