import pytest
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklego.metrics import subset_metric


def test_subset_metric_accuracy_pandas(slicing_classification_dataset):
    accuracy_x1_0 = subset_metric(lambda X, y_true: X['x1'] == 0, accuracy_score)
    accuracy_x1_1 = subset_metric(lambda X, y_true: X['x1'] == 1, accuracy_score)
    X, y = slicing_classification_dataset
    model = RandomForestClassifier().fit(X, y)
    assert accuracy_x1_0(estimator=model, X=X, y_true=y) == 0.5
    assert accuracy_x1_1(estimator=model, X=X, y_true=y) == 1


def test_subset_metric_accuracy_numpy(slicing_classification_dataset):
    accuracy_x1_0 = subset_metric(lambda X, y_true: X[:, 0] == 0, accuracy_score)
    accuracy_x1_1 = subset_metric(lambda X, y_true: X[:, 0] == 1, accuracy_score)
    X, y = slicing_classification_dataset
    X = X.values
    model = RandomForestClassifier().fit(X, y)
    assert accuracy_x1_0(estimator=model, X=X, y_true=y) == 0.5
    assert accuracy_x1_1(estimator=model, X=X, y_true=y) == 1


def test_warning_is_logged_empty_slice(slicing_classification_dataset):
    X, y = slicing_classification_dataset
    accuracy_x1_0 = subset_metric(lambda X, y_true: X['x1'] == 2, accuracy_score)
    model = RandomForestClassifier().fit(X, y)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        accuracy_x1_0(estimator=model, X=X, y_true=y)
        assert issubclass(w[-1].category, RuntimeWarning)


def test_wrong_subset_dimensions_raise_value_error(slicing_classification_dataset):
    X, y = slicing_classification_dataset
    X = X.values
    accuracy_x1_0 = subset_metric(lambda X, y_true: X == 1, accuracy_score)
    model = RandomForestClassifier().fit(X, y)
    with pytest.raises(ValueError):
        accuracy_x1_0(estimator=model, X=X, y_true=y)
