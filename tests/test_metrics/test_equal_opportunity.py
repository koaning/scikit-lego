import types

import pandas as pd
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklego.metrics import equal_opportunity_score
from sklego.preprocessing import ColumnSelector


def test_equal_opportunity_pandas():
    sensitive_classification_dataset = pd.DataFrame(
        {
            "x1": [1, 0, 1, 0, 1, 0, 1, 1],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1],
            "y": [1, 1, 1, 0, 1, 0, 0, 1],
        }
    )

    X, y = (
        sensitive_classification_dataset.drop(columns="y"),
        sensitive_classification_dataset["y"],
    )

    mod_1 = types.SimpleNamespace()

    mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 1, 1])
    assert equal_opportunity_score(sensitive_column="x2")(mod_1, X, y) == 0.75

    mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 1])
    assert equal_opportunity_score(sensitive_column="x2")(mod_1, X, y) == 0.75

    mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 0])
    assert equal_opportunity_score(sensitive_column="x2")(mod_1, X, y) == 0


def test_p_percent_pandas_multiclass():
    sensitive_classification_dataset = pd.DataFrame(
        {
            "x1": [1, 0, 1, 0, 1, 0, 1, 1],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1],
            "y": [1, 1, 1, 0, 1, 0, 0, 2],
        }
    )

    X, y = (
        sensitive_classification_dataset.drop(columns="y"),
        sensitive_classification_dataset["y"],
    )

    mod_1 = types.SimpleNamespace()

    mod_1.predict = lambda X: np.array([2, 0, 1, 0, 1, 0, 1, 2])
    assert (
        equal_opportunity_score(sensitive_column="x2", positive_target=2)(
            mod_1, X, np.array([2, 0, 1, 0, 1, 0, 1, 2])
        )
        == 1
    )

    mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 1])
    assert (
        equal_opportunity_score(sensitive_column="x2", positive_target=2)(mod_1, X, y)
        == 0
    )

    mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 0])
    assert (
        equal_opportunity_score(sensitive_column="x2", positive_target=2)(mod_1, X, y)
        == 0
    )


def test_p_percent_numpy(sensitive_classification_dataset):
    X, y = sensitive_classification_dataset
    X = X.values
    mod = LogisticRegression().fit(X, y)
    assert equal_opportunity_score(1)(mod, X, y) == 0


def test_warning_is_logged(sensitive_classification_dataset):
    X, y = sensitive_classification_dataset
    mod_fair = make_pipeline(ColumnSelector("x1"), LogisticRegression()).fit(X, y)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        equal_opportunity_score("x2", positive_target=2)(mod_fair, X, y)
        assert issubclass(w[-1].category, RuntimeWarning)
