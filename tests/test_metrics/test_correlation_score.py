import pandas as pd
from sklearn.linear_model import Ridge

from sklego.metrics import correlation_score


def test_corr_pandas():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [0, 0, 0, 1, 0, 0, 0, 0],
            "y": [2, 3, 4, 6, 6, 7, 8, 9],
        }
    )

    mod = Ridge().fit(df[["x1", "x2"]], df["y"])
    assert abs(correlation_score("x1")(mod, df[["x1", "x2"]])) > abs(0.99)
    assert abs(correlation_score("x2")(mod, df[["x1", "x2"]])) < abs(0.02)


def test_corr_numpy():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [0, 0, 0, 1, 0, 0, 0, 0],
            "y": [2, 3, 4, 6, 6, 7, 8, 9],
        }
    )
    arr = df[["x1", "x2"]].values
    mod = Ridge().fit(arr, df["y"])
    assert abs(correlation_score(0)(mod, arr)) > abs(0.99)
    assert abs(correlation_score(1)(mod, arr)) < abs(0.02)
