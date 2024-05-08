import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.decomposition import PCAOutlierDetection


@parametrize_with_checks([PCAOutlierDetection(n_components=2, threshold=0.05, random_state=42, variant="absolute")])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 10))])


def test_obvious_usecase(dataset):
    mod = PCAOutlierDetection(n_components=2, threshold=2.5, random_state=42, variant="absolute").fit(dataset)
    assert mod.predict(
        [
            [
                10,
            ]
            * 10
        ]
    ) == np.array([-1])
    assert mod.predict([[0.01] * 10]) == np.array([1])
