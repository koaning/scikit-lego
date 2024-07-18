import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.naive_bayes import BayesianGaussianMixtureNB, GaussianMixtureNB


@parametrize_with_checks(
    [
        GaussianMixtureNB(),
        GaussianMixtureNB(n_components=2),
        BayesianGaussianMixtureNB(),
        BayesianGaussianMixtureNB(n_components=2),
    ],
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 2))])


@pytest.mark.parametrize("k", [1, 5, 10])
def test_obvious_usecase(k):
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (GaussianMixtureNB(n_components=k, max_iter=1000).fit(X, y).predict(X) == y).all()
    assert (BayesianGaussianMixtureNB(n_components=k, max_iter=1000).fit(X, y).predict(X) == y).all()
