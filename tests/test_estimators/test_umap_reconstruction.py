import pytest
import numpy as np

from sklego.common import flatten
from sklego.decomposition import UMAPOutlierDetection
from sklearn.utils import estimator_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            # non-meta checks
            estimator_checks.check_estimators_dtypes,
            estimator_checks.check_fit_score_takes_y,
            estimator_checks.check_dtype_object,
            estimator_checks.check_sample_weights_pandas_series,
            estimator_checks.check_sample_weights_list,
            estimator_checks.check_sample_weights_invariance,
            estimator_checks.check_estimators_fit_returns_self,
            estimator_checks.check_complex_data,
            estimator_checks.check_estimators_empty_data_messages,
            estimator_checks.check_pipeline_consistency,
            estimator_checks.check_estimators_nan_inf,
            estimator_checks.check_estimators_overwrite_params,
            estimator_checks.check_estimator_sparse_data,
            estimator_checks.check_estimators_pickle,
            # general checks
            # estimator_checks.check_fit2d_predict1d -> overrides n_components=1 which is illegal here
            # estimator_checks.check_methods_subset_invariance -> overrides n_components=1 which is illegal here
            # estimator_checks.check_fit2d_1sample -> overrides n_components=1 which is illegal here
            # estimator_checks.check_fit2d_1feature -> overrides n_components=1 which is illegal here
            estimator_checks.check_fit1d,
            estimator_checks.check_get_params_invariance,
            estimator_checks.check_set_params,
            # estimator_checks.check_dict_unchanged -> overrides n_components=1 which is illegal here
            # estimator_checks.check_dont_overwrite_parameters -> overrides n_components=1 which is illegal here
            # outlier_checks
            estimator_checks.check_outliers_fit_predict,
            # estimator_checks.check_outliers_train -> umap doesn't work with score_sampels/decision_func
            # estimator_checks.check_classifier_data_not_an_array -> umap does not deal with this edge case
            estimator_checks.check_estimators_unfitted,
        ]
    ),
)
def test_estimator_checks(test_fn):
    outlier_mod = UMAPOutlierDetection(n_components=2, threshold=0.1)
    test_fn(UMAPOutlierDetection.__name__, outlier_mod)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 10))])


def test_obvious_usecase(dataset):
    mod = UMAPOutlierDetection(n_components=2, threshold=7.5, random_state=42, variant='absolute').fit(dataset)
    assert mod.predict([[10] * 10]) == np.array([-1])
    assert mod.predict([[0.01] * 10]) == np.array([1])
