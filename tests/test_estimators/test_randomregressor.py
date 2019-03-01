from sklego.dummy import RandomRegressor

import pytest


def test_values_uniform(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    mod = RandomRegressor(strategy="uniform")
    predictions = mod.fit(X, y).predict(X)
    assert (predictions >= y.min()).all()
    assert (predictions <= y.max()).all()
    assert mod.min_ == pytest.approx(y.min(), abs=0.0001)
    assert mod.max_ == pytest.approx(y.max(), abs=0.0001)


def test_values_normal(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    mod = RandomRegressor(strategy="normal")
    predictions = mod.fit(X, y).predict(X)
    assert mod.mu_ == y.mean()
    assert mod.sigma_ == y.std()


def test_bad_values():
    with pytest.raises(ValueError):
        RandomRegressor(strategy="foobar")