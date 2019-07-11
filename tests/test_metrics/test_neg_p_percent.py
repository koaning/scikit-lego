import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklego.metrics import p_percent_score
from sklego.preprocessing import ColumnSelector


@pytest.fixture
def sensitive_classification_dataset():
    df = pd.DataFrame({"x1": [1, 0, 1, 0, 1, 0, 1, 1],
                       "x2": [0, 0, 0, 0, 0, 1, 1, 1],
                       "y": [1, 1, 1, 0, 1, 0, 0, 0]})

    return df[['x1', 'x2']], df['y']


@pytest.fixture
def sensitive_multiclass_classification_dataset():
    df = pd.DataFrame({
        'x1': [1, 0, 1, 0, 1, 0, 1, 1, -2, -2, -2, -2],
        'x2': [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        'y': [1, 1, 1, 0, 1, 0, 0, 0, 2, 2, 0, 0],
    })
    return df[['x1', 'x2']], df['y']


def test_p_percent_pandas(sensitive_classification_dataset):
    X, y = sensitive_classification_dataset
    mod_unfair = LogisticRegression().fit(X, y)
    assert p_percent_score("x2")(mod_unfair, X) == 0

    mod_fair = make_pipeline(
        ColumnSelector('x1'),
        LogisticRegression(),
    ).fit(X, y)
    assert p_percent_score("x2")(mod_fair, X) == 0.9


def test_p_percent_pandas_multiclass(sensitive_multiclass_classification_dataset):
    X, y = sensitive_multiclass_classification_dataset
    mod_unfair = LogisticRegression(multi_class='ovr').fit(X, y)
    assert p_percent_score("x2")(mod_unfair, X) == 0
    assert p_percent_score("x2", positive_target=2)(mod_unfair, X) == 0

    mod_fair = make_pipeline(
        ColumnSelector('x1'),
        LogisticRegression(),
    ).fit(X, y)
    assert p_percent_score("x2")(mod_fair, X) == pytest.approx(0.9333333)
    assert p_percent_score("x2", positive_target=2)(mod_fair, X) == 1


def test_p_percent_numpy(sensitive_classification_dataset):
    X, y = sensitive_classification_dataset
    X = X.values
    mod = LogisticRegression().fit(X, y)
    assert p_percent_score(1)(mod, X) == 0
