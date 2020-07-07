import pytest
from sklego.datasets import load_chicken, load_abalone, make_simpleseries, load_hearts, load_penguins


def test_chickweight1():
    X, y = load_chicken(return_X_y=True)
    assert X.shape == (578, 3)
    assert y.shape[0] == 578


def test_chickweight2():
    df = load_chicken(as_frame=True)
    assert df.shape == (578, 4)


def test_abalone1():
    X, y = load_abalone(return_X_y=True)
    assert X.shape == (4177, 8)
    assert y.shape[0] == 4177


def test_abalone2():
    df = load_abalone(as_frame=True)
    assert df.shape == (4177, 9)


def test_simpleseries_constant_season():
    df = (
        make_simpleseries(
            n_samples=365 * 2,
            as_frame=True,
            start_date="2018-01-01",
            trend=0,
            noise=0,
            season_trend=0,
        )
        .assign(month=lambda d: d["date"].dt.month)
        .assign(year=lambda d: d["date"].dt.year)
    )
    agg = df.groupby(["year", "month"]).mean().reset_index()
    assert agg.loc[lambda d: d["month"] == 1].var().month == pytest.approx(
        0.0, abs=0.01
    )


def test_load_hearts():
    df = load_hearts(as_frame=True)
    assert df.shape == (303, 14)


def test_penguin1():
    X, y = load_penguins(return_X_y=True)
    assert X.shape == (344, 6)
    assert y.shape[0] == 344


def test_penguin2():
    df = load_penguins(as_frame=True)
    assert df.shape == (344, 7)
