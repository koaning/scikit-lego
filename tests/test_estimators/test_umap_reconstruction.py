import pytest

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
            estimator_checks.check_fit2d_predict1d,
            estimator_checks.check_methods_subset_invariance,
            estimator_checks.check_fit2d_1sample,
            estimator_checks.check_fit2d_1feature,
            estimator_checks.check_fit1d,
            estimator_checks.check_get_params_invariance,
            estimator_checks.check_set_params,
            estimator_checks.check_dict_unchanged,
            estimator_checks.check_dont_overwrite_parameters,
            # outlier_checks
            estimator_checks.check_outliers_fit_predict,
            estimator_checks.check_outliers_train,
            estimator_checks.check_classifier_data_not_an_array,
            estimator_checks.check_estimators_unfitted,
        ]
    ),
)
def test_estimator_checks(test_fn):
    outlier_mod = UMAPOutlierDetection(n_components=2, threshold=0.1)
    test_fn(UMAPOutlierDetection.__name__, outlier_mod)


def test_buggy():
    outlier_mod = UMAPOutlierDetection(n_components=2, threshold=0.1)
    estimator_checks.check_outliers_train(UMAPOutlierDetection.__name__, outlier_mod)
