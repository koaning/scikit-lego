import numpy as np
import pytest

from sklego.common import flatten
from sklego.preprocessing import IdentityTransformer

from tests.conftest import select_tests, transformer_checks, general_checks, nonmeta_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, transformer_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
def test_estimator_checks(test_fn):
    test_fn(IdentityTransformer.__name__, IdentityTransformer(check_X=True))


def test_same_values(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    X_new = IdentityTransformer(check_X=True).fit_transform(X)
    assert np.isclose(X, X_new).all()


def test_nan_inf(random_xy_dataset_regr):
    # see https://github.com/koaning/scikit-lego/pull/527
    X, y = random_xy_dataset_regr
    X = X.astype(np.float32)
    X[np.random.ranf(size=X.shape) > 0.9] = np.nan
    X[np.random.ranf(size=X.shape) > 0.9] = -np.inf
    X[np.random.ranf(size=X.shape) > 0.9] = np.inf
    X_new = IdentityTransformer(check_X=False).fit_transform(X)


def test_get_feature_names_in(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    it = IdentityTransformer()

    # get_feature_names_out should not work without given input_features if IdentityTransformer is not fitted.
    with pytest.raises(ValueError):
        it.get_feature_names_out(input_features=None)

    # Test with no input_features after being fitted
    it.fit(X, y)
    feature_names = it.get_feature_names_out()
    expected_feature_names = [f"x{i}" for i in range(X.shape[1])]
    np.testing.assert_array_equal(feature_names, expected_feature_names)
