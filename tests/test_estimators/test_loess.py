import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from sklego.linear import LoessRegressor


def test_loessregressor_init_normal():
    """
    Checks initializing the LoessRegressor in normal use cases
    """

    assert LoessRegressor(span=.6)
    assert LoessRegressor(span=1)


def test_loessregressor_init_errors():
    """
    Checks initializing the LoessRegressor with bad settings throw ValueErrors.
    """
    xs = np.linspace(start=0, stop=9, num=10).reshape(-1, 1)
    ys = np.random.random(size=10)

    with pytest.raises(ValueError):
        LoessRegressor(span=0).fit(xs, ys)
    with pytest.raises(ValueError):
        LoessRegressor(span=2).fit(xs, ys)

    with pytest.raises(ValueError):
        LoessRegressor(span=0.5, weighting_method='test').fit(xs, ys)


def test_loessregressor_fit_normal():
    """
    Checks fitting the LoessRegressor returns expected values in normal use cases
    """
    x = np.random.random(size=10).reshape(-1, 1)
    y = np.random.random(size=10)

    model = LoessRegressor(span=0.5).fit(x, y)
    assert all((x == model.xs).reshape(-1, 1))
    assert all((y == model.ys).reshape(-1, 1))


def test_loessregressor_fit_errors():
    """
    Checks fitting the LoessRegressor with bad values throw ValueErrors.
    """
    with pytest.raises(ValueError):
        x = np.array([]).reshape(-1, 1)
        y = np.random.random(size=10)
        LoessRegressor(span=0.5).fit(x, y)

    with pytest.raises(ValueError):
        x = np.random.random(size=10).reshape(-1, 1)
        y = np.array([])
        LoessRegressor(span=0.5).fit(x, y)

    with pytest.raises(ValueError):
        x = np.random.random(size=2).reshape(-1, 1)
        y = np.random.random(size=9)
        LoessRegressor(span=0.5).fit(x, y)


def test_loessregressor_get_window_indices_normal():
    """
    Checks the window indices are returned as expected.
    """
    xs = np.linspace(start=0, stop=9, num=10).reshape(-1, 1)
    ys = np.random.random(size=10)

    # 100% window
    model = LoessRegressor(span=1).fit(xs, ys)
    assert not set(xs.flatten()) - set(model._get_window_indices(xs))

    # Exactly 1 element windows
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


def test_loessregressor_in_pipeline():
    """
    Check the LoessRegressor doesn't break an sklearn pipeline object in calling fit and predict
    methods and returns expected values
    """

    model = LoessRegressor(span=0.5)

    pipeline = Pipeline([('loess', model)])

    xs = np.linspace(start=0, stop=9, num=10)
    ys = np.array([x * 2 for x in xs])

    y_preds_expected = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]

    y_preds = pipeline.fit(xs.reshape(-1, 1), ys).predict(xs.reshape(-1, 1))
    y_preds = [np.round(y, decimals=3) for y in y_preds]

    assert y_preds == y_preds_expected


