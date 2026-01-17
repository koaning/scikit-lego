import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.meta import ConfusionBalancer


@parametrize_with_checks(
    [
        ConfusionBalancer(estimator=LogisticRegression(solver="lbfgs"), alpha=alpha, cfm_smooth=cfm_smooth)
        for alpha in (0.1, 0.5, 0.9)
        for cfm_smooth in (0, 1, 2)
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_sum_equals_one():
    np.random.seed(42)
    n1, n2 = 100, 500
    X = np.concatenate([np.random.normal(0, 1, (n1, 2)), np.random.normal(2, 1, (n2, 2))], axis=0)
    y = np.concatenate([np.zeros((n1, 1)), np.ones((n2, 1))], axis=0).reshape(-1)
    mod = ConfusionBalancer(
        LogisticRegression(solver="lbfgs", max_iter=1000),
        alpha=0.1,
    )
    mod.fit(X, y)
    assert np.all(np.isclose(mod.predict_proba(X).sum(axis=1), 1, atol=0.001))
