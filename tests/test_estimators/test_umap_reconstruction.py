import pytest
import numpy as np

from sklego.common import flatten
from sklego.decomposition import UMAPOutlierDetection
from tests.conftest import general_checks, outlier_checks, select_tests, nonmeta_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, outlier_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_outliers_fit_predict",
            "check_outliers_train",
            "check_fit2d_predict1d",
            "check_methods_subset_invariance",
            "check_fit2d_1sample",
            "check_fit2d_1feature",
            "check_dict_unchanged",
            "check_dont_overwrite_parameters",
            "check_classifier_data_not_an_array",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
def test_estimator_checks(test_fn):
    outlier_mod = UMAPOutlierDetection(n_components=2, threshold=0.1)
    test_fn(UMAPOutlierDetection.__name__, outlier_mod)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (200, 10))])


def test_obvious_usecase(dataset):
    mod = UMAPOutlierDetection(n_components=2, threshold=7.5, random_state=42, variant='absolute').fit(dataset)
    assert mod.predict([[10] * 10]) == np.array([-1])
    assert mod.predict([[0.01] * 10]) == np.array([1])
