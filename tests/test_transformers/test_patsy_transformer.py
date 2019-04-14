import pytest
import numpy as np
import pandas as pd

from sklego.transformers import PatsyTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture()
def df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5, 6],
                         "b": np.log([10, 9, 8, 7, 6, 5]),
                         "c": ["a", "b", "a", "b", "c", "c"],
                         "d": ["b", "a", "a", "b", "a", "b"],
                         "e": [0, 1, 0, 1, 0, 1]})


def test_basic_usage(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = PatsyTransformer("a + b")
    assert tf.fit(X, y).transform(X).shape == (6, 3)


def test_min_sign_usage(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = PatsyTransformer("a + b - 1")
    assert tf.fit(X, y).transform(X).shape == (6, 2)


def test_apply_numpy_transform(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = PatsyTransformer("a + np.log(a) + b - 1")
    assert tf.fit(X, y).transform(X).shape == (6, 3)


def test_multiply_columns(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = PatsyTransformer("a*b - 1")
    print(tf.fit(X, y).transform(X))
    assert tf.fit(X, y).transform(X).shape == (6, 3)


def test_transform_dummy1(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = PatsyTransformer("a + b + d")
    print(tf.fit(X, y).transform(X))
    assert tf.fit(X, y).transform(X).shape == (6, 4)


def test_transform_dummy2(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = PatsyTransformer("a + b + c + d")
    print(tf.fit(X, y).transform(X))
    assert tf.fit(X, y).transform(X).shape == (6, 6)


def test_mult_usage(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]]
    tf = PatsyTransformer("a*b - 1")
    print(tf.fit(X, y).transform(X))
    assert tf.fit(X, y).transform(X).shape == (6, 3)


def test_design_matrix_in_pipeline(df):
    X, y = df[["a", "b", "c", "d"]], df[["e"]].values.ravel()
    pipe = Pipeline([
        ("design", PatsyTransformer("a + np.log(a) + b - 1")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(solver='lbfgs')),
    ])
    assert pipe.fit(X, y).predict(X).shape == (6,)


def test_subset_categories_in_test(df):
    df_train = df[:5]
    X_train, y_train = df_train[["a", "b", "c", "d"]], df_train[["e"]].values.ravel()

    df_test = df[5:]
    X_test, _ = df_test[["a", "b", "c", "d"]], df_test[["e"]].values.ravel()

    trf = PatsyTransformer("a + np.log(a) + b + c + d - 1")

    trf.fit(X_train, y_train)

    assert trf.transform(X_test).shape[1] == trf.transform(X_train).shape[1]


def test_design_matrix_error(df):
    df_train = df[:4]
    X_train, y_train = df_train[["a", "b", "c", "d"]], df_train[["e"]].values.ravel()

    df_test = df[4:]
    X_test, _ = df_test[["a", "b", "c", "d"]], df_test[["e"]].values.ravel()

    pipe = Pipeline([
        ("design", PatsyTransformer("a + np.log(a) + b + c + d - 1")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(solver='lbfgs')),
    ])

    pipe.fit(X_train, y_train)
    with pytest.raises(RuntimeError):
        pipe.predict(X_test)
