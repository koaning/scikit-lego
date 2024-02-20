from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklego.common import flatten
from sklego.feature_selection.mrmr import MaximumRelevanceMinimumRedundancy, _redundancy_pearson
from tests.conftest import general_checks, transformer_checks


@pytest.mark.parametrize("test_fn", flatten([general_checks, transformer_checks]))
def test_transformer_checks(test_fn):
    mrmr = MaximumRelevanceMinimumRedundancy(k=1)
    test_fn(MaximumRelevanceMinimumRedundancy.__name__, mrmr)


def test_redundancy_pearson():
    """Test the pearson correlation for selected and left features:
    For the 3rd column (index 2), it mirrors the values in the first and second columns (indexes 0, 1),
    resulting in a correlation of 1 between the 3rd column and both the 1st and 2nd columns.
    For the 4th column, the correlation with the 1st and 2nd columns is |(-1)|, which equals 1.
    The total correlation sum is 4, accounting for all correlations.
    """
    X = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])

    selected = [0, 1]
    left = [2, 3]
    redundancy_scores = _redundancy_pearson(X, selected, left)
    redundancy_scores_expected = np.array([2 * 1.0, 2 * 1.0])
    assert np.allclose(redundancy_scores, redundancy_scores_expected, atol=0.00001)


@pytest.mark.parametrize("k", [1.2, 5, -1])
def test_mrmr_raises(k):
    X, y = make_classification(n_features=4)
    with pytest.raises(ValueError):
        _ = MaximumRelevanceMinimumRedundancy(k=k).fit(X, y)


@pytest.mark.parametrize(
    "dataset",
    [make_classification(n_features=4), make_regression(n_features=4), make_classification(n_features=10)],
)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_mrmr_fit(dataset, k):
    X, y = dataset
    mrmr = MaximumRelevanceMinimumRedundancy(k=k).fit(X, y)
    mask = mrmr._get_support_mask()

    assert np.sum(mask) == k
    assert mask.size == mrmr.n_features_in_ == X.shape[1]


@pytest.mark.parametrize(
    "dataset",
    [make_classification(n_features=4), make_classification(n_features=10)],
)
@pytest.mark.parametrize("k", [1, 3])
def test_mrmr_sklearn_compatible(dataset, k):
    X_c, y_c = dataset
    pipeline = Pipeline(
        [
            ("mrmr", MaximumRelevanceMinimumRedundancy(k=k, kind="auto", redundancy_func="p", relevance_func="f")),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    pipeline = pipeline.fit(X_c, y_c)
    mrmr = pipeline.named_steps["mrmr"]
    rf = pipeline.named_steps["rf"]

    assert len(mrmr.selected_features_) == k == rf.n_features_in_


@pytest.mark.parametrize("relevance, redundancy", [("f", "c"), ("a", "p")])
def test_raises_wrong_relevance_redundancy_parameters(relevance, redundancy):
    X, y = make_classification(n_features=4)
    with pytest.raises(ValueError):
        _ = MaximumRelevanceMinimumRedundancy(k=2, redundancy_func=redundancy, relevance_func=relevance).fit(X, y)


# Forcing to have boolean types on the y -> Fails the kind auto determination
@pytest.mark.parametrize(
    "kind, dataset",
    [("auto", [i.astype(bool) if n == 1 else i for n, i in enumerate(make_classification(n_features=4))])],
)
def test_raises_wrong_kind_parameters(kind, dataset):
    X, y = dataset
    with pytest.raises(ValueError):
        _ = MaximumRelevanceMinimumRedundancy(k=2, kind=kind).fit(X, y)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_mrmr_count_n_of_calls(k):
    X, y = make_classification(n_features=10)
    relevance_mock, redundancy_mock = MagicMock(), MagicMock()

    # MagicMocking the functions to only match the signature of the original functions
    relevance_mock.side_effect = lambda x, y: np.ones(len(x))
    redundancy_mock.side_effect = lambda p1, p2, p3: np.ones(len(p3))

    _ = MaximumRelevanceMinimumRedundancy(k=k, redundancy_func=redundancy_mock, relevance_func=relevance_mock).fit(X, y)
    relevance_mock.assert_called_once()
    assert redundancy_mock.call_count == k - 1
