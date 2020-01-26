import pytest

from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.preprocessing import IntervalEncoder
from tests.conftest import transformer_checks, general_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            transformer_checks,
            general_checks,
            estimator_checks.check_estimators_dtypes,
            estimator_checks.check_fit_score_takes_y,
            # estimator_checks.check_dtype_object,
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
        ]
    ),
)
def test_estimator_checks(test_fn):
    test_fn(IntervalEncoder.__name__, IntervalEncoder(n_chunks=2))


def test_shape_remains_same(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    x_transform = IntervalEncoder(n_chunks=2).fit(X, y).transform(X)
    assert x_transform.shape == X.shape
