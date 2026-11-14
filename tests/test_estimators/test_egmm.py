import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.mixture import EGaussianMixture
from sklego.mixture.egmm import egmm_alpha, egmm_batch_size


@parametrize_with_checks(
    [
        EGaussianMixture(n_components=2, random_state=42),
        EGaussianMixture(n_components=2, covariance_type="spherical", random_state=42),
        EGaussianMixture(n_components=2, batch_size=8, alpha=0.5, init="random", random_state=0),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=4_000, centers=8, n_features=5, cluster_std=0.6, random_state=42)
    return X[:3_000], X[3_000:], y[:3_000]


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_density_matches_sklearn_with_the_same_parameters(blobs, covariance_type):
    X_train, X_test, _ = blobs
    model = EGaussianMixture(n_components=8, covariance_type=covariance_type, random_state=0).fit(X_train)
    reference = GaussianMixture(n_components=8, covariance_type=covariance_type)
    reference.weights_ = model.weights_
    reference.means_ = model.means_
    reference.covariances_ = model.covariances_
    reference.precisions_cholesky_ = model.precisions_cholesky_

    np.testing.assert_allclose(model.score_samples(X_test), reference.score_samples(X_test), rtol=1e-10)
    np.testing.assert_allclose(model.predict_proba(X_test), reference.predict_proba(X_test), atol=1e-10)
    np.testing.assert_array_equal(model.predict(X_test), reference.predict(X_test))
    assert model.score(X_test) == pytest.approx(reference.score(X_test))
    assert model.bic(X_test) == pytest.approx(reference.bic(X_test))
    assert model.aic(X_test) == pytest.approx(reference.aic(X_test))
    np.testing.assert_allclose(model.precisions_, 1.0 / model.covariances_)


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_as_good_as_em(blobs, covariance_type):
    X_train, X_test, y_train = blobs
    model = EGaussianMixture(n_components=8, covariance_type=covariance_type, random_state=0).fit(X_train)
    em = GaussianMixture(n_components=8, covariance_type=covariance_type, random_state=0).fit(X_train)
    assert model.score(X_test) > em.score(X_test) - 0.05
    assert adjusted_rand_score(y_train, model.predict(X_train)) > 0.95


def test_far_from_the_origin(blobs):
    """Data is centred internally, so a large offset does not break the variances."""
    X_train, X_test, _ = blobs
    near = EGaussianMixture(n_components=8, random_state=0).fit(X_train)
    far = EGaussianMixture(n_components=8, random_state=0).fit(X_train + 1e6)
    np.testing.assert_allclose(far.means_ - 1e6, near.means_, atol=1e-6)
    np.testing.assert_allclose(far.covariances_, near.covariances_, rtol=1e-4)
    assert far.score(X_test + 1e6) == pytest.approx(near.score(X_test), abs=1e-4)


@pytest.mark.parametrize("init", ["e-means", "k-means++", "random"])
@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_init_options_and_shapes(blobs, init, covariance_type):
    X_train, _, _ = blobs
    model = EGaussianMixture(n_components=8, covariance_type=covariance_type, init=init, random_state=1)
    model.fit(X_train)
    covariance_shape = (8, 5) if covariance_type == "diag" else (8,)
    assert model.means_.shape == (8, 5)
    assert model.covariances_.shape == covariance_shape
    assert model.precisions_cholesky_.shape == covariance_shape
    assert model.weights_.sum() == pytest.approx(1.0)
    assert np.all(model.covariances_ > 0)


def test_init_array_is_reproducible(blobs):
    X_train, _, _ = blobs
    init = X_train[:8]
    a = EGaussianMixture(n_components=8, init=init, n_epochs=2, random_state=3).fit(X_train)
    b = EGaussianMixture(n_components=8, init=init, n_epochs=2, random_state=3).fit(X_train)
    np.testing.assert_allclose(a.means_, b.means_)
    np.testing.assert_allclose(a.covariances_, b.covariances_)


def test_fitted_attributes(blobs):
    X_train, _, _ = blobs
    model = EGaussianMixture(n_components=8, n_epochs=2.5, batch_size=256, random_state=0).fit(X_train)
    assert model.batch_size_ == 256
    assert model.alpha_ == pytest.approx(egmm_alpha(3_000, 8, 2.5, 256))
    assert model.n_iter_ == 3
    assert model.n_steps_ == 2 * 12 + 6
    assert np.isfinite(model.lower_bound_)
    assert model.n_features_in_ == 5


def test_default_batch_size_and_alpha():
    assert egmm_batch_size(320_000, 256, 5.0, 100) == 16_384
    assert egmm_batch_size(100, 8, 5.0, 2) == 100
    assert egmm_alpha(320_000, 256, 5.0, 16_384) == 0.3
    assert 0 < egmm_alpha(3_000, 8, 1.0, 512) <= 0.3


def test_sample(blobs):
    X_train, _, _ = blobs
    for covariance_type in ["diag", "spherical"]:
        model = EGaussianMixture(n_components=8, covariance_type=covariance_type, random_state=0).fit(X_train)
        samples, labels = model.sample(20_000)
        assert samples.shape == (20_000, 5)
        assert labels.shape == (20_000,)
        np.testing.assert_allclose(np.bincount(labels, minlength=8) / 20_000, model.weights_, atol=0.02)
        big = np.argmax(model.weights_)
        np.testing.assert_allclose(samples[labels == big].mean(axis=0), model.means_[big], atol=0.1)


def test_fit_predict(blobs):
    X_train, _, _ = blobs
    model = EGaussianMixture(n_components=8, random_state=0)
    np.testing.assert_array_equal(model.fit_predict(X_train), model.predict(X_train))


@pytest.mark.parametrize(
    "params",
    [
        {"n_components": 0},
        {"covariance_type": "full"},
        {"n_epochs": 0},
        {"batch_size": 0},
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"reg_covar": -1.0},
        {"init": "kmeans"},
        {"init": np.zeros((3, 5))},
        {"unused_threshold": -1},
        {"backend": "cuda"},
    ],
)
def test_invalid_params(blobs, params):
    X_train, _, _ = blobs
    with pytest.raises(ValueError):
        EGaussianMixture(**{"n_components": 8, **params}).fit(X_train)


def test_sample_needs_positive_n(blobs):
    X_train, _, _ = blobs
    model = EGaussianMixture(n_components=2, random_state=0).fit(X_train)
    with pytest.raises(ValueError):
        model.sample(0)


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_revival_fixes_collapsed_init(covariance_type):
    """Components that all start at one point share every row; revival splits them apart, as in `EMeans`."""
    X, y = make_blobs(n_samples=4_000, centers=8, n_features=4, cluster_std=0.3, random_state=0)
    init = np.repeat(X[:1], 8, axis=0) + np.random.default_rng(0).normal(scale=1e-6, size=(8, 4))
    params = dict(n_components=8, covariance_type=covariance_type, init=init, random_state=0)
    revived = EGaussianMixture(**params).fit(X)
    stuck = EGaussianMixture(**params, unused_threshold=0).fit(X)
    assert revived.n_splits_ > 0 and stuck.n_splits_ == 0
    assert revived.score(X) > stuck.score(X) + 1.0
    assert adjusted_rand_score(y, revived.predict(X)) > adjusted_rand_score(y, stuck.predict(X)) + 0.2


def test_no_revival_with_two_features():
    X, _ = make_blobs(n_samples=2_000, centers=4, n_features=2, random_state=0)
    assert EGaussianMixture(n_components=4, random_state=0).fit(X).n_splits_ == 0
