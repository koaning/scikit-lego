import itertools as it

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import estimator_checks

n_vals = (10, 500)
k_vals = (1, 5)
np_types = (np.int32, np.float32, np.float64)

transformer_checks = (
    estimator_checks.check_transformer_data_not_an_array,
    estimator_checks.check_transformer_general,
    estimator_checks.check_transformers_unfitted,
)

general_checks = (
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_methods_subset_invariance,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit2d_1feature,
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters,
)

nonmeta_checks = (
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
)

classifier_checks = (
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_classifiers_one_label,
    estimator_checks.check_classifiers_classes,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_classifiers_train,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_estimators_unfitted,
    estimator_checks.check_non_transformer_estimators_n_iter,
    estimator_checks.check_decision_proba_consistency,
)

regressor_checks = (
    estimator_checks.check_regressors_train,
    estimator_checks.check_regressor_data_not_an_array,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_regressors_no_decision_function,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_regressors_int,
    estimator_checks.check_estimators_unfitted,
)

outlier_checks = (
    estimator_checks.check_outliers_fit_predict,
    estimator_checks.check_outliers_train,
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_estimators_unfitted,
)


def select_tests(include, exclude=[]):
    """Return an iterable of include with all tests whose name is not in exclude"""
    for test in include:
        if test.__name__ not in exclude:
            yield test


@pytest.fixture(
    scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)]
)
def random_xy_dataset_regr(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,))
    return X, y


@pytest.fixture(
    scope="module", params=[_ for _ in it.product([10, 100], [1, 2, 3], np_types)]
)
def random_xy_dataset_regr_small(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,))
    return X, y


@pytest.fixture(
    scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)]
)
def random_xy_dataset_clf(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.normal(0, 2, (n,)) > 0.0
    return X, y


@pytest.fixture(
    scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)]
)
def random_xy_dataset_multiclf(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = pd.cut(np.random.normal(0, 2, (n,)), 3).codes
    return X, y


@pytest.fixture(
    scope="module", params=[_ for _ in it.product(n_vals, k_vals, np_types)]
)
def random_xy_dataset_multitarget(request):
    n, k, np_type = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k)).astype(np_type)
    y = np.random.randint(0, 2, (n, k)) > 0.0
    return X, y


@pytest.fixture
def sensitive_classification_dataset():
    df = pd.DataFrame(
        {
            "x1": [1, 0, 1, 0, 1, 0, 1, 1],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1],
            "y": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )

    return df[["x1", "x2"]], df["y"]


@pytest.fixture
def sensitive_multiclass_classification_dataset():
    df = pd.DataFrame(
        {
            "x1": [1, 0, 1, 0, 1, 0, 1, 1, -2, -2, -2, -2],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            "y": [1, 1, 1, 0, 1, 0, 0, 0, 2, 2, 0, 0],
        }
    )
    return df[["x1", "x2"]], df["y"]


def id_func(param):
    """Returns the repr of an object for usage in pytest parametrize"""
    return repr(param)
