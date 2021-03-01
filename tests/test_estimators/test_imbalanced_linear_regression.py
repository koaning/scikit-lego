"""Test the ImbalancedLinearRegression."""

import numpy as np
import pytest

from sklego.linear_model import ImbalancedLinearRegression
from sklego.testing import check_shape_remains_same_regressor
from sklego.common import flatten
from tests.conftest import general_checks, nonmeta_checks, select_tests, regressor_checks

test_batch = [
    (np.array([0, 0, 3, 0, 6]), 3),
    (np.array([1, 0, -2, 0, 4, 0, -5, 0, 6]), 2),
    (np.array([4, -4]), 0),
]


def _create_dataset(coefs, intercept, noise=0.0):
    np.random.seed(0)
    X = np.random.randn(1000, coefs.shape[0])
    y = X @ coefs + intercept + noise * np.random.randn(1000)

    return X, y


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise(coefs, intercept):
    """Regression problems without noise."""
    X, y = _create_dataset(coefs, intercept)
    imb = ImbalancedLinearRegression()
    imb.fit(X, y)

    assert imb.score(X, y) > 0.99


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_score(coefs, intercept):
    """Tests with noise on an easy problem. A good score should be possible."""
    X, y = _create_dataset(coefs, intercept, noise=0.1)
    imb = ImbalancedLinearRegression()
    imb.fit(X, y)

    assert imb.score(X, y) > 0.9


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise_positive(coefs, intercept):
    """Test with only positive coefficients."""
    X, y = _create_dataset(coefs, intercept)
    imb = ImbalancedLinearRegression(positive=True)
    imb.fit(X, y)

    assert all(imb.coef_ >= 0)
    assert imb.score(X, y) > 0.5


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise_regularization(coefs, intercept):
    """Test model with regularization. The size of the coef vector should shrink the larger alpha gets."""
    X, y = _create_dataset(coefs, intercept)
    imbs = [
        ImbalancedLinearRegression(alpha=alpha, l1_ratio=0.5).fit(X, y)
        for alpha in range(4)
    ]
    coef_size = np.array([np.sum(imb.coef_ ** 2) for imb in imbs])

    for i in range(3):
        assert coef_size[i] >= coef_size[i + 1]


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_under_estimation(coefs, intercept):
    """Test if the model is able to underestimate."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    imb = ImbalancedLinearRegression(overestimation_punishment_factor=50)
    imb.fit(X, y)

    assert (imb.predict(X) < y).mean() > 0.8


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_over_estimation(coefs, intercept):
    """Test if the model is able to overestimate."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    imb = ImbalancedLinearRegression(overestimation_punishment_factor=0.01)
    imb.fit(X, y)

    assert (imb.predict(X) < y).mean() < 0.15


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_fit_intercept_and_copy(coefs, intercept):
    """Test if fit_intercept and copy_X work."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    imb = ImbalancedLinearRegression(fit_intercept=False, copy_X=False)
    imb.fit(X, y)

    assert imb.intercept_ == 0.0


@pytest.mark.parametrize("test_fn", [check_shape_remains_same_regressor])
def test_imbalanced(test_fn):
    regr = ImbalancedLinearRegression()
    test_fn(ImbalancedLinearRegression.__name__, regr)

@pytest.mark.parametrize(
    "regr", [
         (ImbalancedLinearRegression.__name__, ImbalancedLinearRegression()),
         (ImbalancedLinearRegression.__name__ + "_positive", ImbalancedLinearRegression(positive=True)),
         (ImbalancedLinearRegression.__name__ + "_positive__no_intercept", ImbalancedLinearRegression(positive=True, fit_intercept=False)),
         (ImbalancedLinearRegression.__name__ + "_no_intercept", ImbalancedLinearRegression(fit_intercept=False))
     ]
)
@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, regressor_checks]),
    )
)
def test_estimator_checks(regr, test_fn):
    test_fn(*regr)