import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from sklego.meta import DecayEstimator
from sklego.meta._decay_utils import stepwise_decay


@pytest.mark.parametrize("n_samples,n_steps", [(10, 6), (10, 9), (13, 8), (5, 4)])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.2, 0.8)])
def test_nondivisible_step_counts_stay_within_requested_bounds(n_samples, n_steps, bounds):
    min_value, max_value = bounds
    weights = stepwise_decay(None, np.zeros(n_samples), n_steps=n_steps, min_value=min_value, max_value=max_value)

    assert np.isfinite(weights).all()
    assert (weights >= min_value - 1e-15).all()
    assert (weights <= max_value + 1e-15).all()
    assert (np.diff(weights) >= 0).all()
    assert weights[0] == pytest.approx(min_value)


def test_nondivisible_step_count_spreads_steps_over_all_samples():
    weights = stepwise_decay(None, np.zeros(10), n_steps=6)

    np.testing.assert_allclose(weights, np.array([0, 1, 2, 2, 3, 3, 4, 5, 5, 6]) / 6)


def test_explicit_step_size_is_preserved_when_sample_count_is_not_divisible():
    weights = stepwise_decay(None, np.zeros(10), step_size=6)

    np.testing.assert_allclose(weights, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


@pytest.mark.parametrize("kwargs", [{"n_steps": 4}, {"step_size": 3}])
def test_divisible_steps_preserve_existing_weights(kwargs):
    weights = stepwise_decay(None, np.zeros(12), **kwargs)

    np.testing.assert_allclose(weights, [0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1, 1])


def test_nondivisible_steps_work_in_decay_estimator():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2 * X.ravel() + 3
    model = DecayEstimator(LinearRegression(), decay_func="stepwise", decay_kwargs={"n_steps": 6}).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y)
    assert (model.weights_ >= 0).all()
