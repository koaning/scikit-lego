import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import IdentityTransformer


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
