import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.cluster import EMeans
from sklego.cluster.emeans import _epoch_rows, _scheduled_alpha, emeans_alpha, emeans_batch_size


@parametrize_with_checks([EMeans(n_clusters=3, random_state=42), EMeans(n_clusters=3, batch_size=4, alpha=0.5)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def blobs():
    return make_blobs(n_samples=3_000, centers=10, n_features=8, cluster_std=0.5, random_state=42)


def test_recovers_blobs(blobs):
    X, y = blobs
    model = EMeans(n_clusters=10, init="k-means++", random_state=42).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.95
    np.testing.assert_array_equal(model.labels_, model.predict(X))
    assert model.score(X) == pytest.approx(-model.inertia_)


def test_competitive_with_lloyd():
    X = np.random.default_rng(0).normal(size=(5_000, 16))
    emeans = EMeans(n_clusters=32, n_epochs=3, random_state=0).fit(X)
    lloyd = KMeans(n_clusters=32, init="random", n_init=1, max_iter=25, random_state=0).fit(X)
    assert emeans.inertia_ < 1.01 * lloyd.inertia_


def test_transform_matches_inertia(blobs):
    X, _ = blobs
    model = EMeans(n_clusters=10, random_state=42).fit(X)
    distances = model.transform(X)
    assert distances.shape == (X.shape[0], 10)
    assert (distances.min(axis=1) ** 2).sum() == pytest.approx(model.inertia_)


@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_init_options(blobs, init):
    X, _ = blobs
    assert EMeans(n_clusters=10, init=init, random_state=1).fit(X).cluster_centers_.shape == (10, 8)


def test_init_array_and_alpha_one_is_reproducible(blobs):
    X, _ = blobs
    init = X[:10]
    a = EMeans(n_clusters=10, init=init, alpha=1.0, batch_size=128, random_state=3).fit(X)
    b = EMeans(n_clusters=10, init=init, alpha=1.0, batch_size=128, random_state=3).fit(X)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
    assert a.batch_size_ == 128 and a.alpha_ == 1.0


def test_fitted_attributes(blobs):
    X, _ = blobs
    model = EMeans(n_clusters=10, n_epochs=2.5, batch_size=500, random_state=0).fit(X)
    assert model.n_iter_ == 3
    assert model.n_steps_ == 6 + 3 + 6
    assert len(model.objective_history_) == 3


def test_revival_fixes_collapsed_init():
    X, _ = make_blobs(n_samples=2_000, centers=8, n_features=4, cluster_std=0.3, random_state=0)
    init = np.repeat(X[:1], 8, axis=0) + np.random.default_rng(0).normal(scale=1e-6, size=(8, 4))
    model = EMeans(n_clusters=8, init=init, random_state=0).fit(X)
    assert model.n_splits_ > 0
    assert len(np.unique(model.labels_)) == 8


@pytest.mark.parametrize(
    "params",
    [
        {"n_clusters": 0},
        {"n_epochs": 0},
        {"batch_size": 0},
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"batch_law": "fast"},
        {"init": "kmeans"},
        {"init": np.zeros((2, 8))},
        {"unused_threshold": -1},
        {"n_epochs": 1e-6},
        {"backend": "cuda"},
    ],
)
def test_invalid_params(blobs, params):
    X, _ = blobs
    with pytest.raises(ValueError):
        EMeans(**{"n_clusters": 10, **params}).fit(X)


def test_scaling_laws_match_faiss():
    # Values printed by faiss NumpyEmeans for its README example.
    batch = emeans_batch_size(200_000, 500, 5.0, n_features=64)
    assert batch == 16384
    assert emeans_alpha(200_000, 500, 5.0, batch) == 1.0
    assert emeans_batch_size(10_000, 20, 2.0, n_features=8, batch_law="throughput") == 8 * emeans_batch_size(
        10_000, 20, 2.0, n_features=8, batch_law="quality"
    )


def test_schedule_helpers():
    assert _epoch_rows(10, 3.0) == [10, 10, 10]
    assert _epoch_rows(10, 2.5) == [10, 5, 10]
    assert _epoch_rows(10, 0.5) == [5]
    assert _epoch_rows(5, 1.5) == [2, 5]
    assert _scheduled_alpha(0.5, 0, 100) == 0.5
    assert _scheduled_alpha(0.5, 100, 100) == pytest.approx(0.05)


def test_chunked_assignment_matches_a_single_block(blobs, monkeypatch):
    X, _ = blobs
    model = EMeans(n_clusters=10, random_state=42).fit(X)
    labels, distances, inertia = model.predict(X), model.transform(X), model.score(X)
    monkeypatch.setattr("sklego.cluster.emeans._CHUNK_ELEMENTS", 37)
    np.testing.assert_array_equal(model.predict(X), labels)
    np.testing.assert_allclose(model.transform(X), distances)
    assert model.score(X) == pytest.approx(inertia)
    np.testing.assert_array_equal(EMeans(n_clusters=10, random_state=42).fit(X).labels_, model.labels_)
