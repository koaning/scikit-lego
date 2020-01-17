import pytest
import warnings

import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklego.preprocessing import ColumnSelector

from sklego.metrics import subset_score


class DisabledCV:
    """
    DisabledCV is a helper class that can be used for methods and objects requiring crossvalidation
    where you do not want to actually do crossvalidation. For testing purposes we can use this to
    remove any unpredictability
    """

    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


@pytest.fixture
def slicing_classification_dataset():
    df = pd.DataFrame(
        {
            "x1": [1, 1, 1, 1, 0, 0, 0, 0],
            "x2": [2, 2, 3, 3, 4, 4, 5, 5],
            "y": [0, 1, 1, 1, 0, 0, 0, 1],
        }
    )
    return df[["x1", "x2"]], df["y"]


def test_subset_score_accuracy_pandas(slicing_classification_dataset):
    X, y = slicing_classification_dataset
    model = DummyClassifier(strategy="constant", constant=1).fit(X, y)

    accuracy_x1_0 = subset_score(lambda X, y_true: X["x1"] == 0, accuracy_score)
    accuracy_x1_1 = subset_score(lambda X, y_true: X["x1"] == 1, accuracy_score)
    assert accuracy_x1_0(estimator=model, X=X, y_true=y) == 0.25
    assert accuracy_x1_1(estimator=model, X=X, y_true=y) == 0.75


def test_subset_score_accuracy_numpy(slicing_classification_dataset):
    X, y = slicing_classification_dataset
    X = X.values
    model = DummyClassifier(strategy="constant", constant=1).fit(X, y)

    accuracy_x1_0 = subset_score(lambda X, y_true: X[:, 0] == 0, accuracy_score)
    accuracy_x1_1 = subset_score(lambda X, y_true: X[:, 0] == 1, accuracy_score)
    assert accuracy_x1_0(estimator=model, X=X, y_true=y) == 0.25
    assert accuracy_x1_1(estimator=model, X=X, y_true=y) == 0.75


def test_warning_is_logged_empty_slice(slicing_classification_dataset):
    X, y = slicing_classification_dataset
    model = DummyClassifier(strategy="constant", constant=1).fit(X, y)

    accuracy_x1_0 = subset_score(lambda X, y_true: X["x1"] == 2, accuracy_score)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        accuracy_x1_0(estimator=model, X=X, y_true=y)
        assert issubclass(w[-1].category, RuntimeWarning)


def test_wrong_subset_dimensions_raise_value_error(slicing_classification_dataset):
    X, y = slicing_classification_dataset
    X = X.values
    model = DummyClassifier(strategy="constant", constant=1).fit(X, y)

    accuracy_x1_0 = subset_score(lambda X, y_true: X == 0, accuracy_score)
    with pytest.raises(ValueError):
        accuracy_x1_0(estimator=model, X=X, y_true=y)


def test_subset_score_pipeline(slicing_classification_dataset):
    X, y = slicing_classification_dataset
    model = make_pipeline(
        ColumnSelector("x1"), DummyClassifier(strategy="constant", constant=1)
    ).fit(X, y)

    accuracy_x1_0 = subset_score(lambda X, y_true: X["x1"] == 0, accuracy_score)
    assert accuracy_x1_0(estimator=model, X=X, y_true=y) == 0.25


def test_subset_score_gridsearch(slicing_classification_dataset):
    param_grid = {
        "dummyclassifier__strategy": ["constant"],
        "dummyclassifier__constant": [1],
    }
    pipeline = make_pipeline(DummyClassifier())
    accuracy_x1_0 = subset_score(lambda X, y_true: X["x1"] == 0, accuracy_score)

    cv = DisabledCV()
    search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring=accuracy_x1_0, cv=cv
    )

    X, y = slicing_classification_dataset
    search.fit(X, y)

    assert search.best_score_ == 0.25
