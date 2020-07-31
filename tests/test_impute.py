import itertools as it

import numpy as np
import pytest

from sklego.common import flatten
from sklego.impute import SVDImputer

from tests.conftest import n_vals, k_vals
from tests.conftest import nonmeta_checks, general_checks, transformer_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([nonmeta_checks, transformer_checks, general_checks]),
        exclude=[
            # Fixing nans is what we want
            "check_estimators_nan_inf",
            # Transforming subsets is by design different than designing all at once
            "check_methods_subset_invariance",
            # Nonsense checks because we always need at least two columns (group and value)
            # "check_fit1d",
            # "check_fit2d_1feature",
            # "check_transformer_data_not_an_array",
        ],
    ),
)
def test_estimator_checks(test_fn):
    clf = SVDImputer()
    test_fn(SVDImputer.__name__, clf)


@pytest.fixture(
    scope="module", params=[_ for _ in it.product(n_vals, k_vals, (0.1, 0.2, 0.5))]
)
def random_x_with_missings(request):
    n, k, p = request.param
    np.random.seed(42)
    X = np.random.normal(0, 2, (n, k))
    mask_indices = np.random.choice([True, False], size=X.shape, p=(p, 1 - p))
    X[mask_indices] = np.nan
    return X


@pytest.fixture(
    scope="module", params=[_ for _ in it.product(n_vals, (1, ), (0.01, 0.05, 0.1))]
)
def random_x_collinear(request):
    n, factor, p = request.param
    np.random.seed(42)
    X1 = np.random.normal(0, 2, n)
    X2 = factor * X1

    mask_indices = np.random.choice([True, False], size=X2.shape, p=(p, 1 - p))
    X2[mask_indices] = np.nan

    return np.stack([X1, X2], axis=1), factor


def test_get_kth_approximation_shape(random_xy_dataset_regr):
    X, _ = random_xy_dataset_regr
    for k in range(X.shape[1]):
        svdi = SVDImputer(k)
        assert svdi._get_kth_approximation(X).shape == X.shape


def test_get_kth_approximation_value(random_xy_dataset_regr):
    X, _ = random_xy_dataset_regr
    svdi = SVDImputer(X.shape[1])

    if X.dtype == np.float32:
        # Default error margin to small for float32 dtype
        assert np.allclose(svdi._get_kth_approximation(X), X, atol=1.e-5)
    else:
        assert np.allclose(svdi._get_kth_approximation(X), X, rtol=1.e-4)


def test_nonmissings_unchanged(random_x_with_missings):
    X = random_x_with_missings
    non_missing_idx = np.where(~np.isnan(X))

    k_rank = max(X.shape[1] - 1, 1)

    svdi = SVDImputer(k_rank, use_train=False)

    transformed = svdi.fit_transform(X)

    assert (X[non_missing_idx] == transformed[non_missing_idx]).all()


@pytest.mark.skip
def test_colinear_columns_close(random_x_collinear):
    # TODO: Fix this
    X, factor = random_x_collinear
    svdi = SVDImputer(2, use_train=False)

    X_transformed = svdi.fit_transform(X)

    assert np.allclose(X[:, 0] * factor, X_transformed[:, 1])


def test_fit_transform(random_xy_dataset_regr):
    X, _ = random_xy_dataset_regr

    k_rank = max(X.shape[1] - 1, 1)

    svdi_1 = SVDImputer(k_rank, use_train=False)
    svdi_2 = SVDImputer(k_rank, use_train=True)

    assert np.allclose(svdi_1.fit(X).transform(X), svdi_2.fit_transform(X))
    assert svdi_1.X_ is None
