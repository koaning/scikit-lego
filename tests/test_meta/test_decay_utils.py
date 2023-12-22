from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from sklego.meta._decay_utils import exponential_decay, linear_decay, sigmoid_decay, stepwise_decay


@pytest.mark.parametrize(
    "kwargs, context",
    [
        ({"min_value": 0.1, "max_value": 0.9}, does_not_raise()),
        ({"min_value": 0.1, "max_value": 10}, does_not_raise()),
        ({"min_value": 0.5, "max_value": 0.1}, pytest.raises(ValueError)),
        ({"min_value": "abc", "max_value": 0.1}, pytest.raises(TypeError)),
    ],
)
def test_linear_decay(kwargs, context):
    X, y = np.random.randn(100, 10), np.random.randn(100)

    with context:
        weights = linear_decay(X, y, **kwargs)
        assert np.all(weights[:-1] <= weights[1:])


@pytest.mark.parametrize(
    "kwargs, context",
    [
        ({"decay_rate": 0.9}, does_not_raise()),
        ({"decay_rate": 0.1}, does_not_raise()),
        ({"decay_rate": -1.0}, pytest.raises(ValueError)),
        ({"decay_rate": 2.0}, pytest.raises(ValueError)),
        ({"decay_rate": "abc"}, pytest.raises(TypeError)),
    ],
)
def test_exponential_decay(kwargs, context):
    X, y = np.random.randn(100, 10), np.random.randn(100)

    with context:
        weights = exponential_decay(X, y, **kwargs)
        assert np.all(weights[:-1] <= weights[1:])


@pytest.mark.parametrize(
    "kwargs, context",
    [
        ({"min_value": 0.1, "max_value": 0.9, "n_steps": 10}, does_not_raise()),
        ({"n_steps": 10}, does_not_raise()),
        ({"step_size": 5}, does_not_raise()),
        ({"min_value": 0.5, "max_value": 0.1, "n_steps": 10}, pytest.raises(ValueError)),
        ({"min_value": "abc", "max_value": 0.1, "n_steps": 10}, pytest.raises(TypeError)),
        ({"n_steps": None, "step_size": None}, pytest.raises(ValueError)),
        ({"n_steps": 10, "step_size": 10}, pytest.raises(ValueError)),
        ({"n_steps": 200}, pytest.raises(ValueError)),
        ({"step_size": 200}, pytest.raises(ValueError)),
        ({"n_steps": -2}, pytest.raises(ValueError)),
        ({"step_size": -2}, pytest.raises(ValueError)),
        ({"n_steps": 2.5}, pytest.raises(TypeError)),
        ({"step_size": 2.5}, pytest.raises(TypeError)),
    ],
)
def test_stepwise_decay(kwargs, context):
    X, y = np.random.randn(100, 10), np.random.randn(100)

    with context:
        weights = stepwise_decay(X, y, **kwargs)
        assert np.all(weights[:-1] <= weights[1:])


@pytest.mark.parametrize(
    "kwargs, context",
    [
        ({"min_value": 0.1, "max_value": 0.9}, does_not_raise()),
        ({"min_value": 0.5, "max_value": 0.1}, pytest.raises(ValueError)),
        ({"growth_rate": 0.1}, does_not_raise()),
        ({"growth_rate": -0.1}, pytest.raises(ValueError)),
        ({"growth_rate": 1.1}, pytest.raises(ValueError)),
        ({"abc": 1.1}, pytest.raises(TypeError)),
    ],
)
def test_sigmoid_decay(kwargs, context):
    X, y = np.random.randn(100, 10), np.random.randn(100)

    with context:
        weights = sigmoid_decay(X, y, **kwargs)
        assert np.all(weights[:-1] <= weights[1:])
