import pytest

# from sklearn.dummy import DummyRegressor
# from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression

# from sklearn.pipeline import Pipeline
from sklego.meta import HierarchicalClassifier, HierarchicalRegressor


@pytest.mark.parametrize(
    "meta",
    [
        HierarchicalClassifier(estimator=LogisticRegression(), groups=0),
        HierarchicalRegressor(estimator=LinearRegression(), groups=0),
    ],
)
def test(meta):
    pass
