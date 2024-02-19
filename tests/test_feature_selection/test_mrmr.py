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


# TODO ADD:
# tests to make it break for regression and integer or any combination
# patch red and relevance and count the number of times they are called
