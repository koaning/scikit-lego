import pytest
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklego.metrics import equal_opportunity_score
from sklego.preprocessing import ColumnSelector


def test_equal_opportunity_pandas(sensitive_classification_dataset):
    X, y = sensitive_classification_dataset
    mod_unfair = LogisticRegression().fit(X, y)
    assert equal_opportunity_score("x2")(mod_unfair, X) == 0

    mod_fair = make_pipeline(
        ColumnSelector('x1'),
        LogisticRegression(),
    ).fit(X, y)
    assert equal_opportunity_score("x2")(mod_fair, X) == 0.9


def test_p_percent_pandas_multiclass(sensitive_multiclass_classification_dataset):
    X, y = sensitive_multiclass_classification_dataset
    mod_unfair = LogisticRegression(multi_class='ovr').fit(X, y)
    assert equal_opportunity_score("x2")(mod_unfair, X) == 0
    assert equal_opportunity_score("x2", positive_target=2)(mod_unfair, X) == 0

    mod_fair = make_pipeline(
        ColumnSelector('x1'),
        LogisticRegression(),
    ).fit(X, y)
    assert equal_opportunity_score("x2")(mod_fair, X) == pytest.approx(0.9333333)
    assert equal_opportunity_score("x2", positive_target=2)(mod_fair, X) == 0


def test_p_percent_numpy(sensitive_classification_dataset):
    X, y = sensitive_classification_dataset
    X = X.values
    mod = LogisticRegression().fit(X, y)
    assert equal_opportunity_score(1)(mod, X) == 0


def test_warning_is_logged(sensitive_classification_dataset):
    X, y = sensitive_classification_dataset
    mod_fair = make_pipeline(
        ColumnSelector('x1'),
        LogisticRegression(),
    ).fit(X, y)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        equal_opportunity_score("x2", positive_target=2)(mod_fair, X)
        assert issubclass(w[-1].category, RuntimeWarning)
