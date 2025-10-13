import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.decomposition import UMAPOutlierDetection

pytestmark = pytest.mark.umap


@parametrize_with_checks([UMAPOutlierDetection(n_components=2, threshold=0.1, n_neighbors=3)])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        # numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
        "check_estimators_pickle",
        # ValueError: Need at least 2-D data
        # in `predict`: np.sum(np.abs(self.umap_.inverse_transform(reduced) - X), axis=1)
        "check_dict_unchanged",
        # Numba 0.62.0 fails on this check, probably temporary issue
        "check_estimators_dtypes",
        "check_f_contigous_array_estimator",
    }:
        pytest.skip()

    check(estimator)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (200, 10))])


def test_obvious_usecase(dataset):
    mod = UMAPOutlierDetection(
        n_components=2,
        threshold=7.5,
        random_state=42,
        variant="absolute",
    ).fit(dataset)
    assert mod.predict([[10] * 10]) == np.array([-1])
    assert mod.predict([[0.0001] * 10]) == np.array([1])
