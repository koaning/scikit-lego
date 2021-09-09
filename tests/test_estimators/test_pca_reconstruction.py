import pytest
import numpy as np

from sklego.common import flatten
from sklego.decomposition import PCAOutlierDetection
from tests.conftest import general_checks, outlier_checks, select_tests, nonmeta_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, outlier_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_outliers_fit_predict",
            "check_outliers_train",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
def test_estimator_checks(test_fn):
    outlier_mod = PCAOutlierDetection(n_components=2, threshold=0.05, random_state=42, variant='absolute')
    test_fn(PCAOutlierDetection.__name__, outlier_mod)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 10))])


def test_obvious_usecase(dataset):
    mod = PCAOutlierDetection(n_components=2, threshold=2.5, random_state=42, variant='absolute').fit(dataset)
    assert mod.predict([[10] * 10]) == np.array([-1])
    assert mod.predict([[0.01] * 10]) == np.array([1])
