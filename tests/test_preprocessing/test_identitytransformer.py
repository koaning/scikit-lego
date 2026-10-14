import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import IdentityTransformer


@pytest.mark.parametrize("container", [list, tuple])
def test_array_like_without_shape(container):
    X = container([["a", None], ["b", np.inf]])
    transformer = IdentityTransformer()
    result = transformer.fit_transform(X)
    assert isinstance(result, container)
    assert all(result[i] is X[i] for i in range(len(X)))
    assert transformer.n_samples_ == 2
    assert transformer.n_features_in_ == 2
    assert transformer.shape_ == (2, 2)


@pytest.mark.parametrize("container", [list, tuple])
def test_array_like_feature_count_and_refit(container):
    transformer = IdentityTransformer().fit(container([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="features"):
        transformer.transform(container([[1, 2, 3]]))
    X = container([[1, 2, 3]])
    result = transformer.fit_transform(X)
    assert isinstance(result, container)
    assert all(result[i] is X[i] for i in range(len(X)))
    assert transformer.shape_ == (1, 3)


@pytest.mark.parametrize("container", [list, tuple])
def test_array_like_with_validation(container):
    X = container([[1, 2], [3, 4]])
    transformer = IdentityTransformer(check_X=True)
    np.testing.assert_array_equal(transformer.fit_transform(X), X)
    assert transformer.shape_ == (2, 2)


@pytest.mark.parametrize("X", [np.ones(3), np.ones((2, 3, 4)), [1, 2, 3]])
def test_invalid_dimensions_are_rejected(X):
    transformer = IdentityTransformer()
    with pytest.raises((ValueError, TypeError, AttributeError)):
        transformer.fit(X)
    transformer.fit(np.array([[1, 2], [3, 4]]))
    with pytest.raises((ValueError, TypeError, AttributeError)):
        transformer.fit(X)


@parametrize_with_checks([IdentityTransformer(check_X=True)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


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
    IdentityTransformer(check_X=False).fit_transform(X)
