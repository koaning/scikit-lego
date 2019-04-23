import itertools as it
from copy import copy

from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import make_classification, make_regression


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_shape_remains_same_regressor(name, regressor):
    """Ensure that the estimator does not change the shape of prediction."""
    seed = 42

    for n_samples, n_feat in it.product((100, 1000, 10000), (2, 5, 10)):
        regr = copy(regressor)
        X, y = make_regression(n_samples=n_samples, n_features=n_feat,
                               n_informative=n_feat, noise=2, random_state=seed)
        pred = regr.fit(X, y).predict(X)
        assert y.shape[0] == pred.shape[0]


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_shape_remains_same_classifier(name, classifier):
    """Ensure that the estimator does not change the shape of prediction."""
    seed = 42

    for n_samples, n_feat in it.product((100, 1000, 10000), (2, 5, 10)):
        clf = copy(classifier)
        X, y = make_classification(n_samples=n_samples, n_features=n_feat,
                                   n_informative=n_feat, n_redundant=0, n_repeated=0,
                                   random_state=seed)
        pred = clf.fit(X, y).predict(X)
        assert y.shape[0] == pred.shape[0]
