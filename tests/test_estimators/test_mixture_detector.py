import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.mixture import BayesianGMMOutlierDetector, GMMOutlierDetector


@parametrize_with_checks(
    [
        GMMOutlierDetector(threshold=0.999, method="quantile"),
        GMMOutlierDetector(threshold=2, method="stddev"),
        BayesianGMMOutlierDetector(threshold=0.999, method="quantile"),
        BayesianGMMOutlierDetector(threshold=2, method="stddev"),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 2))])


@pytest.mark.parametrize("model", [GMMOutlierDetector, BayesianGMMOutlierDetector])
def test_obvious_usecase_quantile(dataset, model):
    mod = model(n_components=2, threshold=0.999, method="quantile").fit(dataset)
    assert mod.predict([[10, 10]]) == np.array([-1])
    assert mod.predict([[0, 0]]) == np.array([1])


@pytest.mark.parametrize("model", [GMMOutlierDetector, BayesianGMMOutlierDetector])
def test_obvious_usecase_stddev(dataset, model):
    mod = model(n_components=2, threshold=2, method="stddev").fit(dataset)
    assert mod.predict([[10, 10]]) == np.array([-1])
    assert mod.predict([[0, 0]]) == np.array([1])


@pytest.mark.parametrize("model", [GMMOutlierDetector, BayesianGMMOutlierDetector])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 10},
        {"threshold": -10},
        {"threshold": -10, "method": "stddev"},
    ],
)
def test_value_error_threshold(dataset, model, kwargs):
    with pytest.raises(ValueError):
        model(**kwargs).fit(dataset)


@pytest.mark.parametrize("model", [GMMOutlierDetector, BayesianGMMOutlierDetector])
def test_thresh_effect_stddev(dataset, model):
    mod1 = model(threshold=0.5, method="stddev").fit(dataset)
    mod2 = model(threshold=1, method="stddev").fit(dataset)
    mod3 = model(threshold=2, method="stddev").fit(dataset)
    n_outliers1 = np.sum(mod1.predict(dataset) == -1)
    n_outliers2 = np.sum(mod2.predict(dataset) == -1)
    n_outliers3 = np.sum(mod3.predict(dataset) == -1)
    assert n_outliers1 > n_outliers2
    assert n_outliers2 > n_outliers3


@pytest.mark.parametrize("model", [GMMOutlierDetector, BayesianGMMOutlierDetector])
def test_thresh_effect_quantile(dataset, model):
    mod1 = model(threshold=0.90, method="quantile").fit(dataset)
    mod2 = model(threshold=0.95, method="quantile").fit(dataset)
    mod3 = model(threshold=0.99, method="quantile").fit(dataset)
    n_outliers1 = np.sum(mod1.predict(dataset) == -1)
    n_outliers2 = np.sum(mod2.predict(dataset) == -1)
    n_outliers3 = np.sum(mod3.predict(dataset) == -1)
    assert n_outliers1 > n_outliers2
    assert n_outliers2 > n_outliers3


@pytest.mark.parametrize("model", [GMMOutlierDetector, BayesianGMMOutlierDetector])
def test_obvious_usecase_github(model):
    # from this bug: https://github.com/koaning/scikit-lego/issues/225 thanks Corrie!
    np.random.seed(42)
    X = np.random.normal(-10, 1, (2000, 2))
    mod = model(n_components=1, threshold=0.99).fit(X)

    df = pd.DataFrame(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "loglik": mod.score_samples(X),
            "prediction": mod.predict(X).astype(str),
        }
    )
    assert df.loc[lambda d: d["prediction"] == "-1"].shape[0] == 20
