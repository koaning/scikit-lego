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
        "check_f_contiguous_array_estimator",
        "check_pipeline_consistency",
    }:
        pytest.skip()

    check(estimator)


def test_obvious_usecase():
    input_data = np.random.normal(0, 1, (200, 10))
    try:
        mod = UMAPOutlierDetection(
            n_components=2,
            threshold=7.5,
            random_state=42,
            variant="absolute",
        ).fit(input_data)
        assert mod.predict(np.random.normal(10, 1, (1, 10))) == np.array([-1])
        assert mod.predict(np.random.normal(0, 0.1, (1, 10))) == np.array([1])
    except ZeroDivisionError:
        # This is an issue with UMAP/numba and can't be fixed on our end
        pytest.skip()
