import pytest
from sklego.datasets import load_chicken, make_simpleseries


def test_chickweight1():
    X, y = load_chicken()
    assert X.shape == (578, 3)
    assert y.shape[0] == 578


def test_chickweight2():
    df = load_chicken(give_pandas=True)
    assert df.shape == (578, 4)


def test_simpleseries_constant_season():
    df = (make_simpleseries(n_samples=365*2, give_pandas=True,
                            start_date="2018-01-01", trend=0,
                            noise=0, season_trend=0)
          .assign(month=lambda d: d['date'].dt.month)
          .assign(year=lambda d: d['date'].dt.year))
    agg = df.groupby(['year', 'month']).mean().reset_index()
    assert agg.loc[lambda d: d['month'] == 1].var().month == pytest.approx(0.0, abs=0.01)
