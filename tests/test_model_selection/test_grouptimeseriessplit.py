import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklego.model_selection import GroupTimeSeriesSplit

GROUPS_COL = "group"


@pytest.fixture
def init_data():
    X = np.random.randint(low=1, high=1000, size=17)
    y = np.random.randint(low=1, high=1000, size=17)
    groups = np.array(
        [2000] * 3
        + [2001]
        + [2002] * 2
        + [2003]
        + [2004] * 5
        + [2005] * 2
        + [2006] * 2
        + [2007]
    )
    return X, y, groups


@pytest.fixture(params=[(2), (3), (4), (5), (6), (7)])
def valid_n_splits(request):
    return request.param


@pytest.fixture
def valid_cv(init_data, valid_n_splits):
    X, y, groups = init_data
    cv = GroupTimeSeriesSplit(valid_n_splits)
    _ = cv.split(X, y, groups)
    return cv


@pytest.mark.parametrize("n_splits", [(1), ("a"), (2.2), ([5]), (-1)])
def test_invalid_n_splits(n_splits):
    with pytest.raises(ValueError):
        GroupTimeSeriesSplit(n_splits=n_splits)


@pytest.mark.parametrize("n_splits", [(10), (20), (30), (100)])
def test_too_large_n_splits(init_data, n_splits):
    X, y, groups = init_data
    cv = GroupTimeSeriesSplit(n_splits=n_splits)
    with pytest.raises(ValueError):
        cv.split(X, y, groups)


def test_error_summary_before_split():
    cv = GroupTimeSeriesSplit(2)
    with pytest.raises(AttributeError):
        cv.summary()


def test_summary_method(valid_cv):
    cv_summary = valid_cv.summary()
    assert (
        isinstance(cv_summary, pd.DataFrame) and not cv_summary.empty
    ), f"{type(cv_summary)}\n{cv_summary}"


def test_n_groups_equal_n_splits_plus_one(valid_cv):
    cv_summary = valid_cv.summary()
    assert (
        cv_summary.loc[:, GROUPS_COL].nunique() == valid_cv.get_n_splits() + 1
    ), f"{cv_summary}"


def test_split_points_chronological(valid_cv):
    splits = valid_cv._best_splits
    assert list(splits) == sorted(splits), f"{splits}"


@pytest.mark.parametrize(
    "n_splits, n_groups", [(4, 251), (5, 101), (6, 31), (7, 31)]
)
def test_user_warning(n_splits, n_groups):
    groups = list(range(n_groups))
    X = np.random.randint(1, 10000, size=len(groups))
    y = np.random.randint(1, 10000, size=len(groups))
    cv = GroupTimeSeriesSplit(n_splits)
    with pytest.warns(UserWarning):
        cv.split(X, y, groups)


@pytest.mark.parametrize(
    "X, y, groups",
    [
        (
            np.random.randint(0, 10, size=10),
            np.random.randint(0, 10, size=10),
            np.random.randint(0, 10, size=11),
        ),
        (
            np.random.randint(0, 10, size=10),
            np.random.randint(0, 10, size=11),
            np.random.randint(0, 10, size=10),
        ),
        (
            np.random.randint(0, 10, size=11),
            np.random.randint(0, 10, size=10),
            np.random.randint(0, 10, size=10),
        ),
        (
            np.random.randint(0, 10, size=9),
            np.random.randint(0, 10, size=10),
            np.random.randint(0, 10, size=11),
        ),
    ],
)
def test_different_shapes_x_y_groups(X, y, groups, valid_n_splits):
    cv = GroupTimeSeriesSplit(valid_n_splits)
    with pytest.raises(ValueError):
        cv.split(X, y, groups)


def test_works_with_only_groups(init_data, valid_n_splits):
    _, _, groups = init_data
    cv = GroupTimeSeriesSplit(valid_n_splits)
    assert cv.split(groups=groups)


def test_grouptimeseriessplit_with_gridsearch(valid_n_splits):
    X_train = np.random.randint(1, 1000, size=100)
    y_train = np.random.randint(1, 1000, size=100)
    groups_train = np.random.randint(0, 10, size=100)

    # X_train, y_train, groups_train = init_data
    X_train = X_train.reshape(-1, 1)
    cv_splits = GroupTimeSeriesSplit(valid_n_splits).split(
        X_train, y_train, groups_train
    )
    Lasso(random_state=0, tol=0.1, alpha=0.8).fit(
        X_train, y_train, groups_train
    )
    pipe = Pipeline([("reg", Lasso(random_state=0, tol=0.1))])
    alphas = [0.1, 0.5, 0.8]
    grid = GridSearchCV(pipe, {"reg__alpha": alphas}, cv=cv_splits)
    grid.fit(X_train, y_train)
    best_C = grid.best_estimator_.get_params()["reg__alpha"]
    assert best_C


def test_amount_of_n_splits(init_data, valid_n_splits):
    X, y, groups = init_data
    cv = GroupTimeSeriesSplit(valid_n_splits)
    assert len(cv.split(X, y, groups)) == valid_n_splits


def test_series_same_output_as_arrays(init_data, valid_n_splits):
    X, y, groups = init_data
    df = pd.DataFrame({"X": X, "y": y, "groups": groups})
    cv1 = GroupTimeSeriesSplit(valid_n_splits)
    cv2 = GroupTimeSeriesSplit(valid_n_splits)
    cv1.split(X, y, groups)
    summary1 = cv1.summary()
    cv2.split(df, df["y"], df["groups"])
    summary2 = cv2.summary()
    pd.testing.assert_frame_equal(summary1, summary2)
