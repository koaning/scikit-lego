"""`EGaussianMixture(backend="rust")`: skipped unless `sklego_rust` is installed (`pip install ./rust`)."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.mixture import EGaussianMixture

pytest.importorskip("sklego_rust")
pytestmark = pytest.mark.rust


@parametrize_with_checks(
    [
        EGaussianMixture(n_components=2, random_state=42, backend="rust"),
        EGaussianMixture(n_components=2, covariance_type="spherical", random_state=42, backend="rust"),
        EGaussianMixture(n_components=2, batch_size=8, alpha=0.5, init="random", random_state=0, backend="rust"),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=4_000, centers=8, n_features=5, cluster_std=0.6, random_state=42)
    return X[:3_000], X[3_000:], y[:3_000]


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_full_batches_match_numpy(blobs, covariance_type):
    """With one batch per epoch, a fixed init and no revival, nothing depends on the random numbers."""
    X_train, _, _ = blobs
    params = dict(
        n_components=8,
        covariance_type=covariance_type,
        n_epochs=3,
        batch_size=len(X_train),
        init=X_train[:8],
        unused_threshold=0,
    )
    numpy = EGaussianMixture(**params).fit(X_train)
    rust = EGaussianMixture(**params, backend="rust").fit(X_train)
    for attribute in ["weights_", "means_", "covariances_", "precisions_cholesky_"]:
        np.testing.assert_allclose(getattr(rust, attribute), getattr(numpy, attribute), rtol=1e-9, atol=1e-12)
    assert rust.lower_bound_ == pytest.approx(numpy.lower_bound_, rel=1e-12)
    assert (rust.n_steps_, rust.n_iter_, rust.n_splits_) == (numpy.n_steps_, numpy.n_iter_, numpy.n_splits_)


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_density_matches_numpy(blobs, covariance_type):
    X_train, X_test, _ = blobs
    model = EGaussianMixture(n_components=8, covariance_type=covariance_type, random_state=0).fit(X_train)
    expected = model.score_samples(X_test), model.predict_proba(X_test), model.predict(X_test)
    model.set_params(backend="rust")
    np.testing.assert_allclose(model.score_samples(X_test), expected[0], rtol=1e-10)
    np.testing.assert_allclose(model.predict_proba(X_test), expected[1], atol=1e-10)
    np.testing.assert_array_equal(model.predict(X_test), expected[2])
    assert model.predict(X_test).dtype == np.intp


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_as_good_as_numpy_on_average(blobs, covariance_type):
    X_train, X_test, y_train = blobs

    def mean_score(backend):
        models = [
            EGaussianMixture(n_components=8, covariance_type=covariance_type, random_state=s, backend=backend)
            for s in range(4)
        ]
        return np.mean([m.fit(X_train).score(X_test) for m in models])

    assert mean_score("rust") > mean_score("numpy") - 0.05


def test_recovers_blobs(blobs):
    X_train, _, y_train = blobs
    model = EGaussianMixture(n_components=8, random_state=0, backend="rust").fit(X_train)
    assert adjusted_rand_score(y_train, model.predict(X_train)) > 0.95


@pytest.mark.parametrize("init", ["e-means", "k-means++", "random"])
@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_init_options_and_shapes(blobs, init, covariance_type):
    X_train, _, _ = blobs
    model = EGaussianMixture(n_components=8, covariance_type=covariance_type, init=init, random_state=1, backend="rust")
    model.fit(X_train)
    covariance_shape = (8, 5) if covariance_type == "diag" else (8,)
    assert model.means_.shape == (8, 5)
    assert model.covariances_.shape == covariance_shape
    assert model.weights_.sum() == pytest.approx(1.0)
    assert np.all(model.covariances_ > 0)


def test_same_seed_same_model(blobs):
    X_train, _, _ = blobs
    a = EGaussianMixture(n_components=8, random_state=3, backend="rust").fit(X_train)
    b = EGaussianMixture(n_components=8, random_state=3, backend="rust").fit(X_train)
    np.testing.assert_array_equal(a.means_, b.means_)
    np.testing.assert_array_equal(a.covariances_, b.covariances_)


def test_far_from_the_origin(blobs):
    X_train, X_test, _ = blobs
    near = EGaussianMixture(n_components=8, random_state=0, backend="rust").fit(X_train)
    far = EGaussianMixture(n_components=8, random_state=0, backend="rust").fit(X_train + 1e6)
    np.testing.assert_allclose(far.means_ - 1e6, near.means_, atol=1e-6)
    np.testing.assert_allclose(far.covariances_, near.covariances_, rtol=1e-4)
    assert far.score(X_test + 1e6) == pytest.approx(near.score(X_test), abs=1e-4)


def test_fitted_attributes(blobs):
    X_train, _, _ = blobs
    model = EGaussianMixture(n_components=8, n_epochs=2.5, batch_size=256, random_state=0, backend="rust")
    model.fit(X_train)
    assert model.n_iter_ == 3
    assert model.n_steps_ == 2 * 12 + 6
    assert np.isfinite(model.lower_bound_)
    np.testing.assert_allclose(model.precisions_, 1.0 / model.covariances_)


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_revival_fixes_collapsed_init(covariance_type):
    X, y = make_blobs(n_samples=4_000, centers=8, n_features=4, cluster_std=0.3, random_state=0)
    init = np.repeat(X[:1], 8, axis=0) + np.random.default_rng(0).normal(scale=1e-6, size=(8, 4))
    params = dict(n_components=8, covariance_type=covariance_type, init=init, random_state=0, backend="rust")
    revived = EGaussianMixture(**params).fit(X)
    stuck = EGaussianMixture(**params, unused_threshold=0).fit(X)
    assert revived.n_splits_ > 0 and stuck.n_splits_ == 0
    assert revived.score(X) > stuck.score(X) + 1.0
    assert adjusted_rand_score(y, revived.predict(X)) > adjusted_rand_score(y, stuck.predict(X)) + 0.2
