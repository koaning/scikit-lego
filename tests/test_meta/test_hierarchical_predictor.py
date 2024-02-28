from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest
from sklearn import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklego.meta import HierarchicalClassifier, HierarchicalRegressor


def make_hierarchical_dataset(task):
    n_samples, n_features, n_informative, random_state = 1000, 10, 3, 42
    if task == "binary-classification":
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state
        )

    elif task == "multiclass-classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=random_state,
            n_classes=4,
        )

    elif task == "regression":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state
        )

    else:
        raise ValueError("Invalid task")

    X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])]).assign(
        g_0=1,
        g_1=["A"] * 500 + ["B"] * 500,
        g_2=["X"] * 250 + ["Y"] * 500 + ["Z"] * 250,
    )
    groups = ["g_0", "g_1", "g_2"]

    return X, y, groups


@pytest.mark.parametrize(
    "meta_cls,base_estimator,task",
    [
        (HierarchicalRegressor, LinearRegression(), "regression"),
        (HierarchicalRegressor, Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), "regression"),
        (HierarchicalClassifier, LogisticRegression(), "binary-classification"),
        (
            HierarchicalClassifier,
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]),
            "binary-classification",
        ),
        (HierarchicalClassifier, LogisticRegression(), "multiclass-classification"),
        (
            HierarchicalClassifier,
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]),
            "multiclass-classification",
        ),
    ],
)
@pytest.mark.parametrize("fallback_method", ["raise", "parent"])
@pytest.mark.parametrize(
    "shrinkage",
    [
        {"shrinkage": None},
        {"shrinkage": "equal"},
        {"shrinkage": "relative"},
        {"shrinkage": "min_n_obs", "min_n_obs": 10},
        {"shrinkage": "constant", "alpha": 0.5},
    ],
)
def test_fit_predict(meta_cls, base_estimator, task, fallback_method, shrinkage):
    """Tests that the model can be fit and predict with different configurations of fallback and shrinkage methods if
    X to predict contains same groups as X used to fit.
    """
    X, y, groups = make_hierarchical_dataset(task)

    meta_model = meta_cls(estimator=base_estimator, groups=groups, fallback_method=fallback_method, **shrinkage).fit(
        X, y
    )

    assert meta_model.estimators_ is not None
    assert meta_model.predict(X) is not None

    if task in {"binary-classification", "multiclass-classification"}:
        assert meta_model.predict_proba(X) is not None


@pytest.mark.parametrize(
    "meta_cls,base_estimator,task",
    [
        (HierarchicalRegressor, LinearRegression(), "regression"),
        (HierarchicalClassifier, LogisticRegression(), "binary-classification"),
        (HierarchicalClassifier, LogisticRegression(), "multiclass-classification"),
    ],
)
@pytest.mark.parametrize("fallback_method,context", [("raise", pytest.raises(KeyError)), ("parent", does_not_raise())])
def test_fallback(meta_cls, base_estimator, task, fallback_method, context):
    """Tests that the model fails or not when predicting with different fallback methods if X to predict contains
    unseen group values.
    """
    X, y, groups = make_hierarchical_dataset(task)

    meta_model = meta_cls(estimator=base_estimator, groups=groups, fallback_method=fallback_method).fit(X, y)
    X.loc[:, groups] = "unseen_group_value"
    with context:
        meta_model.predict(X)


@pytest.mark.parametrize(
    "meta_cls,base_estimator,task,metric",
    [
        (HierarchicalRegressor, LinearRegression(), "regression", r2_score),
        (HierarchicalClassifier, LogisticRegression(), "binary-classification", accuracy_score),
        (HierarchicalClassifier, LogisticRegression(), "multiclass-classification", accuracy_score),
    ],
)
@pytest.mark.parametrize(
    "shrinkage",
    [
        {"shrinkage": None},
        {"shrinkage": "equal"},
        {"shrinkage": "relative"},
        {"shrinkage": "min_n_obs", "min_n_obs": 10},
        {"shrinkage": "constant", "alpha": 0.5},
    ],
)
def test_shrinkage(meta_cls, base_estimator, task, metric, shrinkage):
    """Tests that the model performance is better than the base estimator when predicting with different shrinkage
    methods.
    """
    X, y, groups = make_hierarchical_dataset(task)

    meta_model = meta_cls(estimator=clone(base_estimator), groups=groups, **shrinkage).fit(X, y)
    base_model = clone(base_estimator).fit(X.drop(columns=groups), y)

    assert metric(y, base_model.predict(X.drop(columns=groups))) <= metric(y, meta_model.predict(X))
