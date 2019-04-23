import pytest
from pandas.tests.extension.numpy_.test_numpy_nested import np
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_X_y

from sklego.common import flatten
from sklego.transformers import EstimatorTransformer
from tests.conftest import transformer_checks, nonmeta_checks, general_checks


@pytest.mark.parametrize("test_fn", flatten([
    transformer_checks,
    nonmeta_checks,
    general_checks,
]))
def test_estimator_checks(test_fn):
    trf = EstimatorTransformer(LinearRegression())
    test_fn(EstimatorTransformer.__name__, trf)


def test_values_uniform(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X, y = check_X_y(X, y)
    clf = DummyClassifier(strategy='most_frequent')
    transformer = EstimatorTransformer(clone(clf))
    transformed = transformer.fit(X, y).transform(X)

    assert transformed.shape == (y.shape[0],)
    assert np.all(transformed == clf.fit(X, y).predict(X))


def test_set_params():
    clf = DummyClassifier(strategy='most_frequent')
    transformer = EstimatorTransformer(clf)

    transformer.set_params(estimator__strategy='stratified')
    assert clf.strategy == 'stratified'


def test_get_params():
    clf = DummyClassifier(strategy='most_frequent')
    transformer = EstimatorTransformer(clf)

    assert transformer.get_params() == {
        'estimator__constant': None,
        'estimator__random_state': None,
        'estimator': clf,
        'estimator__strategy': 'most_frequent',
        'predict_func': 'predict'
    }
