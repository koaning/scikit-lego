import numpy as np
import pytest

from sklego.loess import LoessRegressor


def test_loessregressor_init_normal():
    xs = np.linspace(start=0, stop=9, num=10)
    ys = np.random.random(size=10)

    LoessRegressor(span=.6).fit(xs, ys)
    LoessRegressor(span=1).fit(xs, ys)


def test_loessregressor_init_errors():
    xs = np.linspace(start=0, stop=9, num=10)
    ys = np.random.random(size=10)

    with pytest.raises(ValueError):
        LoessRegressor(span=0).fit(xs, ys)
    with pytest.raises(ValueError):
        LoessRegressor(span=2).fit(xs, ys)


def test_loessregressor_fit_normal():
    x = np.random.random(size=10)
    y = np.random.random(size=10)

    model = LoessRegressor(span=0.5).fit(x, y)
    assert (x, y) == (model.xs, model.ys)


def test_loessregressor_fit_errors():
    with pytest.raises(ValueError):
        x = np.array([])
        y = np.random.random(size=10)
        LoessRegressor(span=0.5).fit(x, y)

    with pytest.raises(ValueError):
        x = np.random.random(size=10)
        y = np.array([])
        LoessRegressor(span=0.5).fit(x, y)

    with pytest.raises(ValueError):
        x = np.random.random(size=2)
        y = np.random.random(size=9)
        LoessRegressor(span=0.5).fit(x, y)


def test_loessregressor_get_window_indices_normal():
    xs = np.linspace(start=0, stop=9, num=10)
    ys = np.random.random(size=10)

    # 100% window
    model = LoessRegressor(span=1).fit(xs, ys)
    assert all(xs == model._get_window_indices(xs.reshape(-1, 1)))

    model = LoessRegressor(span=0.1).fit(xs, ys)
    assert all(x == model._get_window_indices(x.reshape(-1, 1)) for x in xs)

    # Partial windows
    model = LoessRegressor(span=0.5).fit(xs, ys)

    expected = {0, 1, 2, 3, 4}
    result = set(model._get_window_indices(xs[0].reshape(-1, 1)))
    assert expected == result

    expected = {1, 2, 3, 4, 5}
    result = set(model._get_window_indices(xs[3].reshape(-1, 1)))
    assert expected == result

    expected = {4, 5, 6, 7, 8}
    result = set(model._get_window_indices(xs[6].reshape(-1, 1)))
    assert expected == result

    expected = {5, 6, 7, 8, 9}
    result = set(model._get_window_indices(xs[-1].reshape(-1, 1)))
    assert expected == result