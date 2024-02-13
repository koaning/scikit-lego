import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklego.feature_selection.mrmr import MaximumRelevanceMinimumRedundancy, _redundancy_pearson


@pytest.fixture
def simple_regression():
    X, y = make_regression(n_features=4)
    return X, y


@pytest.fixture
def simple_classification():
    X, y = make_classification(n_features=4)
    return X, y


def test_redundancy_pearson():
    X = np.array([[0.0, 0.0, 0.0, 4.5], [1.0, 1.0, 1.0, 8.9], [1.0, 1.0, 1.0, 12.3], [1.0, 1.0, 1.0, 16.7]])

    selected = [0, 1]
    left = [2, 3]
    redundancy_scores = _redundancy_pearson(X, selected, left)
    redundancy_scores_expected = np.array([2 * 1.0, 2 * 0.78652])

    assert np.allclose(redundancy_scores, redundancy_scores_expected, atol=0.00001)


def test_mrmr_raises(simple_classification):
    X, y = simple_classification
    with pytest.raises(ValueError):
        _ = MaximumRelevanceMinimumRedundancy(k=1.2).fit(X, y)
        _ = MaximumRelevanceMinimumRedundancy(k=5).fit(X, y)
        _ = MaximumRelevanceMinimumRedundancy(k=-1).fit(X, y)


def test_mrmr_fit(simple_classification, simple_regression):
    X_r, y_r = simple_regression
    X_c, y_c = simple_classification
    mask_r = MaximumRelevanceMinimumRedundancy(k=2).fit(X_r, y_r)._get_support_mask()
    mask_c = MaximumRelevanceMinimumRedundancy(k=2).fit(X_c, y_c)._get_support_mask()

    assert all([np.sum(mask_r) == 2, np.sum(mask_c) == 2])


def test_mrmr_sklearn_compatible(simple_classification):
    X_c, y_c = simple_classification
    pipeline = Pipeline(
        [
            ("mrmr", MaximumRelevanceMinimumRedundancy(k=3, kind="auto", redundancy_func="p", relevance_func="f")),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    pipeline = pipeline.fit(X_c, y_c)
    mrmr = pipeline.named_steps["mrmr"]
    rf = pipeline.named_steps["rf"]

    assert len(mrmr.selected_features_) == 3
    assert rf.n_features_in_ == 3
