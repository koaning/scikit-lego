import pytest
import pandas as pd
import numpy as np

from sklego.preprocessing import OrthogonalTransformer
from sklego.common import flatten

from tests.conftest import nonmeta_checks, general_checks, transformer_checks


@pytest.fixture
def sample_matrix():
    np.random.seed(1313)
    return np.random.normal(size=(50, 10))


@pytest.fixture
def sample_df(sample_matrix):
    return pd.DataFrame(sample_matrix)


@pytest.mark.parametrize(
    "test_fn", flatten([nonmeta_checks, general_checks, transformer_checks])
)
def test_estimator_checks(test_fn):
    test_fn(OrthogonalTransformer.__name__, OrthogonalTransformer())


def check_is_orthogonal(X, tolerance=10 ** -5):
    """
    Check if X is an column orthogonal matrix. If X is column orthogonal, then X.T * X equals the identity matrix
    :param X: Matrix to check
    :param tolerance: Tolerance for difference caused by rounding
    :raises: AssertionError if X is not orthogonal
    """
    diff_with_eye = np.dot(X.T, X) - np.eye(X.shape[1])

    if np.max(np.abs(diff_with_eye)) > tolerance:
        raise AssertionError("X is not orthogonal")


def check_is_orthonormal(X, tolerance=10 ** -5):
    """
    Check if X is an column orthonormal matrix, i.e. orthogonal and with columns with norm 1.
    :param X: Matrix to check
    :param tolerance: Tolerance for difference caused by rounding
    :raises: AssertionError if X is not orthonormal
    """
    # Orthonormal, so orthogonal and columns must be normalized
    check_is_orthogonal(X, tolerance)

    norms = np.linalg.norm(X, ord=2, axis=0)

    if (max(norms) > 1 + tolerance) or (min(norms) < 1 - tolerance):
        raise AssertionError("X is not orthonormal")


def test_orthogonal_transformer(sample_matrix):
    ot = OrthogonalTransformer(normalize=False)
    ot.fit(X=sample_matrix)

    assert hasattr(ot, "inv_R_")
    assert hasattr(ot, "normalization_vector_")
    assert ot.inv_R_.shape[0] == sample_matrix.shape[1]

    assert all(ot.normalization_vector_ == 1)

    trans = ot.transform(sample_matrix)

    check_is_orthogonal(trans)


def test_orthonormal_transformer(sample_matrix):
    ot = OrthogonalTransformer(normalize=True)
    ot.fit(X=sample_matrix)

    assert hasattr(ot, "inv_R_")
    assert hasattr(ot, "normalization_vector_")
    assert ot.inv_R_.shape[0] == sample_matrix.shape[1]
    assert ot.normalization_vector_.shape[0] == sample_matrix.shape[1]

    trans = ot.transform(sample_matrix)

    check_is_orthonormal(trans)


def test_orthogonal_with_df(sample_df):
    ot = OrthogonalTransformer(normalize=False)
    ot.fit(X=sample_df)

    assert ot.inv_R_.shape[0] == sample_df.shape[1]

    trans = ot.transform(sample_df)

    check_is_orthogonal(trans)
