import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal as pandas_assert_frame_equal
from polars.testing import assert_frame_equal as polars_assert_frame_equal
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklego.model_selection import TimeGapSplit

df = pd.DataFrame(np.random.randint(0, 30, size=(30, 4)), columns=list("ABCy"))
df["date"] = pd.date_range(start="1/1/2018", end="1/30/2018")[::-1]
df.index = df.index.astype(str) + "_"
df = df.sort_values("date")

train = df.head(25)

X_train = train[["A", "B", "C"]]
y_train = train["y"]


def test_timegapsplit():
    cv = TimeGapSplit(
        date_serie=df["date"],
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=0),
    )

    for i, indices in enumerate(cv.split(X_train, y_train)):
        train_mindate = df.loc[X_train.iloc[indices[0]].index]["date"].min()
        train_maxdate = df.loc[X_train.iloc[indices[0]].index]["date"].max()
        valid_mindate = df.loc[X_train.iloc[indices[1]].index]["date"].min()
        valid_maxdate = df.loc[X_train.iloc[indices[1]].index]["date"].max()

        assert train_mindate <= train_maxdate <= valid_mindate <= valid_maxdate

    # regression testing, check if output changes of the last fold
    assert train_mindate == datetime.datetime.strptime("2018-01-16", "%Y-%m-%d")
    assert train_maxdate == datetime.datetime.strptime("2018-01-20", "%Y-%m-%d")
    assert valid_mindate == datetime.datetime.strptime("2018-01-21", "%Y-%m-%d")
    assert valid_maxdate == datetime.datetime.strptime("2018-01-23", "%Y-%m-%d")

    expected = [
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7])),
        (np.array([3, 4, 5, 6, 7]), np.array([8, 9, 10])),
        (np.array([6, 7, 8, 9, 10]), np.array([11, 12, 13])),
        (np.array([9, 10, 11, 12, 13]), np.array([14, 15, 16])),
        (np.array([12, 13, 14, 15, 16]), np.array([17, 18, 19])),
        (np.array([15, 16, 17, 18, 19]), np.array([20, 21, 22])),
    ]
    for result_indices, expected_indices in zip(list(cv.split(X_train, y_train)), expected):
        np.testing.assert_array_equal(result_indices[0], expected_indices[0])
        np.testing.assert_array_equal(result_indices[1], expected_indices[1])

    # Polars doesn't have an index, so this class behaves a bit differenly for
    # index-less objects. We need to first ensure that `date_serie`, `X_train`,
    # and `y_train` all have the same length.
    date_serie = df["date"].loc[X_train.index]
    cv = TimeGapSplit(
        date_serie=pl.from_pandas(date_serie),
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=0),
    )
    expected = [
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7])),
        (np.array([3, 4, 5, 6, 7]), np.array([8, 9, 10])),
        (np.array([6, 7, 8, 9, 10]), np.array([11, 12, 13])),
        (np.array([9, 10, 11, 12, 13]), np.array([14, 15, 16])),
        (np.array([12, 13, 14, 15, 16]), np.array([17, 18, 19])),
        (np.array([15, 16, 17, 18, 19]), np.array([20, 21, 22])),
    ]
    for result_indices, expected_indices in zip(
        list(cv.split(pl.from_pandas(X_train), pl.from_pandas(y_train))), expected
    ):
        np.testing.assert_array_equal(result_indices[0], expected_indices[0])
        np.testing.assert_array_equal(result_indices[1], expected_indices[1])


def test_timegapsplit_too_big_gap():
    try:
        TimeGapSplit(
            date_serie=df["date"],
            train_duration=timedelta(days=5),
            valid_duration=timedelta(days=3),
            gap_duration=timedelta(days=5),
        )
    except ValueError:
        print("Successfully failed")


def test_timegapsplit_using_splits():
    cv = TimeGapSplit(
        date_serie=df["date"],
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=1),
        n_splits=3,
    )
    assert len(list(cv.split(X_train, y_train))) == 3


def test_timegapsplit_too_many_splits():
    cv = TimeGapSplit(
        date_serie=df["date"],
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=1),
        n_splits=7,
    )
    with pytest.raises(ValueError):
        list(cv.split(X_train, y_train))


def test_timegapsplit_train_or_nsplit():
    with pytest.raises(ValueError):
        _ = TimeGapSplit(
            date_serie=df["date"],
            train_duration=None,
            valid_duration=timedelta(days=3),
            gap_duration=timedelta(days=5),
            n_splits=None,
        )


def test_timegapsplit_without_train_duration():
    cv = TimeGapSplit(
        date_serie=df["date"],
        train_duration=None,
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=5),
        n_splits=3,
    )
    csv = list(cv.split(X_train, y_train))

    assert len(csv) == 3
    assert cv.train_duration == timedelta(days=10)


def test_timegapsplit_with_a_gap():
    gap_duration = timedelta(days=2)
    cv_gap = TimeGapSplit(
        date_serie=df["date"],
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=gap_duration,
    )

    for i, indices in enumerate(cv_gap.split(X_train, y_train)):
        train_mindate = df.loc[X_train.iloc[indices[0]].index]["date"].min()
        train_maxdate = df.loc[X_train.iloc[indices[0]].index]["date"].max()
        valid_mindate = df.loc[X_train.iloc[indices[1]].index]["date"].min()
        valid_maxdate = df.loc[X_train.iloc[indices[1]].index]["date"].max()

        assert train_mindate <= train_maxdate <= valid_mindate <= valid_maxdate
        assert valid_mindate - train_maxdate >= gap_duration


def test_timegapsplit_with_gridsearch():
    cv = TimeGapSplit(
        date_serie=df["date"],
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=0),
    )

    Lasso(random_state=0, tol=0.1, alpha=0.8).fit(X_train, y_train)

    pipe = Pipeline([("reg", Lasso(random_state=0, tol=0.1))])
    alphas = [0.1, 0.5, 0.8]
    grid = GridSearchCV(pipe, {"reg__alpha": alphas}, cv=cv)
    grid.fit(X_train, y_train)
    best_C = grid.best_estimator_.get_params()["reg__alpha"]

    assert best_C


def test_timegapsplit_summary():
    cv = TimeGapSplit(
        date_serie=df["date"],
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=0),
    )

    summary = cv.summary(X_train)
    assert summary.shape == (12, 5)

    expected_data = {
        "Start date": [
            datetime.datetime(2018, 1, 1, 0, 0),
            datetime.datetime(2018, 1, 6, 0, 0),
            datetime.datetime(2018, 1, 4, 0, 0),
            datetime.datetime(2018, 1, 9, 0, 0),
            datetime.datetime(2018, 1, 7, 0, 0),
            datetime.datetime(2018, 1, 12, 0, 0),
            datetime.datetime(2018, 1, 10, 0, 0),
            datetime.datetime(2018, 1, 15, 0, 0),
            datetime.datetime(2018, 1, 13, 0, 0),
            datetime.datetime(2018, 1, 18, 0, 0),
            datetime.datetime(2018, 1, 16, 0, 0),
            datetime.datetime(2018, 1, 21, 0, 0),
        ],
        "End date": [
            datetime.datetime(2018, 1, 5, 0, 0),
            datetime.datetime(2018, 1, 8, 0, 0),
            datetime.datetime(2018, 1, 8, 0, 0),
            datetime.datetime(2018, 1, 11, 0, 0),
            datetime.datetime(2018, 1, 11, 0, 0),
            datetime.datetime(2018, 1, 14, 0, 0),
            datetime.datetime(2018, 1, 14, 0, 0),
            datetime.datetime(2018, 1, 17, 0, 0),
            datetime.datetime(2018, 1, 17, 0, 0),
            datetime.datetime(2018, 1, 20, 0, 0),
            datetime.datetime(2018, 1, 20, 0, 0),
            datetime.datetime(2018, 1, 23, 0, 0),
        ],
        "Period": [
            datetime.timedelta(days=4),
            datetime.timedelta(days=2),
            datetime.timedelta(days=4),
            datetime.timedelta(days=2),
            datetime.timedelta(days=4),
            datetime.timedelta(days=2),
            datetime.timedelta(days=4),
            datetime.timedelta(days=2),
            datetime.timedelta(days=4),
            datetime.timedelta(days=2),
            datetime.timedelta(days=4),
            datetime.timedelta(days=2),
        ],
        "Unique days": [5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3],
        "nbr samples": [5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3],
        "part": [
            "train",
            "valid",
            "train",
            "valid",
            "train",
            "valid",
            "train",
            "valid",
            "train",
            "valid",
            "train",
            "valid",
        ],
        "fold": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    }
    expected = pd.DataFrame(expected_data).set_index(["fold", "part"])
    pandas_assert_frame_equal(summary, expected)

    # Polars doesn't have an index, so this class behaves a bit differenly for
    # index-less objects. We need to ensure that `date_serie` and `X_train` have
    # the same length.
    date_serie = df["date"].loc[X_train.index]
    cv = TimeGapSplit(
        date_serie=pl.from_pandas(date_serie),
        train_duration=timedelta(days=5),
        valid_duration=timedelta(days=3),
        gap_duration=timedelta(days=0),
    )
    summary = cv.summary(pl.from_pandas(X_train))

    expected = pl.DataFrame(expected_data)
    polars_assert_frame_equal(summary, expected)
