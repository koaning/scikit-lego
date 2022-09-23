"""Test the QuantileRegression."""

import numpy as np
import pytest

from sklego.linear_model import QuantileRegression
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
    X = np.random.randn(10000, coefs.shape[0])
    y = X @ coefs + intercept + noise * np.random.randn(10000)

    return X, y


@pytest.mark.parametrize("method", ["SLSQP", "TNC", "L-BFGS-B"])
@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise(coefs, intercept, method):
    """Regression problems without noise."""
    X, y = _create_dataset(coefs, intercept)
    quant = QuantileRegression(method=method)
    quant.fit(X, y)
    assert quant.score(X, y) > 0.99


@pytest.mark.parametrize("method", ["SLSQP", "TNC", "L-BFGS-B"])
@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_score(coefs, intercept, method):
    """Tests with noise on an easy problem. A good score should be possible."""
    X, y = _create_dataset(coefs, intercept, noise=1.0)
    quant = QuantileRegression(method=method)
    quant.fit(X, y)
    assert quant.score(X, y) > 0.9


@pytest.mark.parametrize("method", ["SLSQP", "TNC", "L-BFGS-B"])
@pytest.mark.parametrize("coefs, intercept", test_batch)
@pytest.mark.parametrize("quantile", np.linspace(0, 1, 7))
def test_quantile(coefs, intercept, quantile, method):
    """Tests with noise on an easy problem. A good score should be possible."""
    X, y = _create_dataset(coefs, intercept, noise=1.0)
    quant = QuantileRegression(method=method, quantile=quantile)
    quant.fit(X, y)

    np.testing.assert_almost_equal((quant.predict(X)>=y).mean(), quantile, decimal=2)


@pytest.mark.parametrize("method", ["SLSQP", "TNC", "L-BFGS-B"])
@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise_positive(coefs, intercept, method):
    """Test with only positive coefficients."""
    X, y = _create_dataset(coefs, intercept, noise=0.0)
    quant = QuantileRegression(method=method, positive=True)
    quant.fit(X, y)
    assert all(quant.coef_ >= 0)
    assert quant.score(X, y) > 0.3


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise_regularization(coefs, intercept):
    """Test model with regularization. The size of the coef vector should shrink the larger alpha gets."""
    X, y = _create_dataset(coefs, intercept)

    quants = [QuantileRegression(alpha=alpha, l1_ratio=0.).fit(X, y) for alpha in range(3)]
    coef_size = np.array([np.sum(quant.coef_ ** 2) for quant in quants])

    for i in range(2):
        assert coef_size[i] >= coef_size[i + 1]


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_fit_intercept_and_copy(coefs, intercept):
    """Test if fit_intercept and copy_X work."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    imb = QuantileRegression(fit_intercept=False, copy_X=False)
    imb.fit(X, y)

    assert imb.intercept_ == 0.0


@pytest.mark.parametrize("test_fn", [check_shape_remains_same_regressor])
def test_quant(test_fn):
    regr = QuantileRegression()
    test_fn(QuantileRegression.__name__, regr)


@pytest.mark.parametrize("quantile", [0.5, 0.3])
@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("method", ["SLSQP", "TNC", "L-BFGS-B"])
@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, regressor_checks]),
    )
)
def test_estimator_checks(positive, fit_intercept, method, quantile, test_fn):
    regr = (
        f'{QuantileRegression.__name__}_quantile_{quantile}_method_{method}_positive_{positive}_fit_intercept_{fit_intercept}',
        QuantileRegression(quantile=quantile, method=method, positive=positive, fit_intercept=fit_intercept)
    )
    test_fn(*regr)
