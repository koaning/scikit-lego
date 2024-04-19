import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklego.preprocessing import FormulaicTransformer

pytestmark = pytest.mark.formulaic


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": np.log([10, 9, 8, 7, 6, 5]),
            "c": ["a", "b", "a", "b", "c", "c"],
            "d": ["b", "a", "a", "b", "a", "b"],
            "e": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize(
    "return_type, expected_type",
    [
        ("numpy", np.ndarray),
        ("pandas", pd.DataFrame),
        ("sparse", spmatrix),
    ],
)
def test_return_type(df, return_type, expected_type):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = FormulaicTransformer("a + b - 1", return_type=return_type)
    df_fit_transformed = tf.fit(X, y).transform(X)
    assert isinstance(df_fit_transformed, expected_type)


@pytest.mark.parametrize(
    "formula, expected_shape",
    [
        ("a + b - 1", (6, 2)),
        ("a + np.log(a) + b - 1", (6, 3)),
        ("a*b - 1", (6, 3)),
        ("a + b + d", (6, 4)),
        ("a + b + c + d", (6, 6)),
    ],
)
def test_formula_output(df, formula, expected_shape):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = FormulaicTransformer(formula=formula)

    assert tf.fit(X, y).transform(X).shape == expected_shape


def test_pipeline(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]].values.ravel()

    pipe = Pipeline(
        [
            ("design", FormulaicTransformer("a + np.log(a) + b - 1")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(solver="lbfgs")),
        ]
    )
    assert pipe.fit(X, y).predict(X).shape[0] == X.shape[0]


def test_unseen_categories(df):
    df_train, df_test = df[:4], df[4:]

    X_train, y_train = df_train[["a", "b", "c", "d"]], df_train[["e"]].values.ravel()
    X_test = df_test[["a", "b", "c", "d"]]

    trf = FormulaicTransformer("a + np.log(a) + b + c + d - 1")
    _ = trf.fit(X_train, y_train)

    assert trf.transform(X_test).shape[1] == trf.transform(X_train).shape[1]

    pipe = Pipeline(
        [
            ("design", FormulaicTransformer("a + np.log(a) + b + c + d - 1")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(solver="lbfgs")),
        ]
    )

    _ = pipe.fit(X_train, y_train)
    assert pipe.predict(X_test).shape[0] == X_test.shape[0]


def test_misshape(df):
    df_train, df_test = df[:4], df[4:]

    X_train, y_train = df_train[["a", "b", "c", "d"]], df_train[["e"]].values.ravel()
    X_test = df_test[["a", "b", "c"]]

    trf = FormulaicTransformer("a + np.log(a) + b + c + d - 1")
    _ = trf.fit(X_train, y_train)

    with pytest.raises(ValueError):
        trf.transform(X_test)


@pytest.mark.parametrize("return_type", ("numpy", "pandas"))
@pytest.mark.parametrize(
    "formula",
    (
        "a + b - 1",
        "a + np.log(a) + b - 1",
        "a*b - 1",
        "a + b + d",
        "a + b + c + d",
    ),
)
def test_pickling(tmp_path, df, return_type, formula):
    df_train, df_test = df[:4], df[4:]

    X_train, y_train = df_train[["a", "b", "c", "d"]], df_train[["e"]].values.ravel()
    X_test = df_test[["a", "b", "c", "d"]]

    pipe = Pipeline(
        [
            ("design", FormulaicTransformer(formula=formula, return_type=return_type)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(solver="lbfgs")),
        ]
    )

    _ = pipe.fit(X_train, y_train)

    joblib.dump(pipe, tmp_path / "pipeline.pkl")
    loaded_pipe = joblib.load(tmp_path / "pipeline.pkl")

    assert loaded_pipe.predict(X_test).shape[0] == X_test.shape[0]
