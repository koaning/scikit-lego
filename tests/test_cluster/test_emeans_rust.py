"""`EMeans(backend="rust")`: skipped unless `sklego_rust` is installed (`pip install ./rust`)."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.cluster import EMeans

pytest.importorskip("sklego_rust")
pytestmark = pytest.mark.rust


@parametrize_with_checks(
    [
        EMeans(n_clusters=3, random_state=42, backend="rust"),
        EMeans(n_clusters=3, batch_size=4, alpha=0.5, backend="rust"),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def blobs():
    return make_blobs(n_samples=3_000, centers=10, n_features=8, cluster_std=0.5, random_state=42)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("alpha", [1.0, 0.4])
def test_full_batches_match_numpy(blobs, dtype, alpha):
    """With one batch per epoch, a fixed init and no revival, nothing depends on the random numbers."""
    X = blobs[0].astype(dtype)
    params = dict(n_clusters=10, n_epochs=3, batch_size=len(X), alpha=alpha, init=X[:10], unused_threshold=0)
    numpy = EMeans(**params).fit(X)
    rust = EMeans(**params, backend="rust").fit(X)
    tolerance = 1e-10 if dtype == np.float64 else 1e-4
    assert rust.cluster_centers_.dtype == dtype
    np.testing.assert_allclose(rust.cluster_centers_, numpy.cluster_centers_, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(rust.labels_, numpy.labels_)
    assert rust.inertia_ == pytest.approx(numpy.inertia_, rel=tolerance)
    np.testing.assert_allclose(rust.objective_history_, numpy.objective_history_, rtol=tolerance)
    assert (rust.n_steps_, rust.n_iter_, rust.n_splits_) == (numpy.n_steps_, numpy.n_iter_, numpy.n_splits_)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_predict_transform_score_match_numpy(blobs, dtype):
    X = blobs[0].astype(dtype)
    model = EMeans(n_clusters=10, random_state=0).fit(X)
    labels, distances, score = model.predict(X), model.transform(X), model.score(X)
    model.set_params(backend="rust")
    tolerance = 1e-10 if dtype == np.float64 else 1e-4
    np.testing.assert_array_equal(model.predict(X), labels)
    np.testing.assert_allclose(model.transform(X), distances, rtol=tolerance, atol=tolerance)
    assert model.transform(X).dtype == dtype
    assert model.score(X) == pytest.approx(score, rel=tolerance)


def test_as_good_as_numpy_on_average():
    X = np.random.default_rng(0).normal(size=(5_000, 16))
    numpy = np.mean([EMeans(n_clusters=32, random_state=s).fit(X).inertia_ for s in range(3)])
    rust = np.mean([EMeans(n_clusters=32, random_state=s, backend="rust").fit(X).inertia_ for s in range(3)])
    assert rust == pytest.approx(numpy, rel=0.01)


def test_recovers_blobs(blobs):
    X, y = blobs
    model = EMeans(n_clusters=10, init="k-means++", random_state=42, backend="rust").fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.95
    np.testing.assert_array_equal(model.labels_, model.predict(X))
    assert model.score(X) == pytest.approx(-model.inertia_)


def test_same_seed_same_model(blobs):
    X, _ = blobs
    a = EMeans(n_clusters=10, random_state=7, backend="rust").fit(X)
    b = EMeans(n_clusters=10, random_state=7, backend="rust").fit(X)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    c = EMeans(n_clusters=10, random_state=8, backend="rust").fit(X)
    assert not np.array_equal(a.cluster_centers_, c.cluster_centers_)


def test_revival_fixes_collapsed_init():
    X, _ = make_blobs(n_samples=2_000, centers=8, n_features=4, cluster_std=0.3, random_state=0)
    init = np.repeat(X[:1], 8, axis=0) + np.random.default_rng(0).normal(scale=1e-6, size=(8, 4))
    model = EMeans(n_clusters=8, init=init, random_state=0, backend="rust").fit(X)
    assert model.n_splits_ > 0
    assert len(np.unique(model.labels_)) == 8


def test_fitted_attributes(blobs):
    X, _ = blobs
    model = EMeans(n_clusters=10, n_epochs=2.5, batch_size=500, random_state=0, backend="rust").fit(X)
    assert model.n_iter_ == 3
    assert model.n_steps_ == 6 + 3 + 6
    assert len(model.objective_history_) == 3
    assert model.labels_.dtype == np.intp


def test_without_scipy_blas_gives_the_same_result(blobs):
    import sklego_rust

    X, _ = blobs
    with_blas = EMeans(n_clusters=10, random_state=0, backend="rust").fit(X)
    sklego_rust.use_blas(False)
    try:
        without = EMeans(n_clusters=10, random_state=0, backend="rust").fit(X)
    finally:
        sklego_rust.use_blas(True)
    np.testing.assert_allclose(without.cluster_centers_, with_blas.cluster_centers_, rtol=1e-8, atol=1e-8)
