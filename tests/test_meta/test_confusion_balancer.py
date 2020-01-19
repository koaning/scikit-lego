import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression


from sklego.common import flatten
from sklego.meta import ConfusionBalancer
from tests.conftest import general_checks, classifier_checks


@pytest.mark.parametrize("test_fn", flatten([general_checks, classifier_checks]))
def test_estimator_checks_classification(test_fn):
    trf = ConfusionBalancer(LogisticRegression(solver="lbfgs"))
    test_fn(ConfusionBalancer.__name__, trf)


def test_sum_equals_one():
    np.random.seed(42)
    n1, n2 = 100, 500
    X = np.concatenate(
        [np.random.normal(0, 1, (n1, 2)), np.random.normal(2, 1, (n2, 2))], axis=0
    )
    y = np.concatenate([np.zeros((n1, 1)), np.ones((n2, 1))], axis=0).reshape(-1)
    mod = ConfusionBalancer(
        LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1000),
        alpha=0.1,
    )
    mod.fit(X, y)
    assert np.all(np.isclose(mod.predict_proba(X).sum(axis=1), 1, atol=0.001))
