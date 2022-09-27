import pytest
import numpy as np
from unittest.mock import Mock
from unittest.mock import patch
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import check_X_y

from sklego.common import flatten
from sklego.meta import EstimatorTransformer
from tests.conftest import transformer_checks, general_checks


@pytest.mark.parametrize(
    "test_fn", flatten([transformer_checks, general_checks])
)
def test_estimator_checks(test_fn):
    trf = EstimatorTransformer(LinearRegression())
    test_fn(EstimatorTransformer.__name__, trf)


def test_values_uniform(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X, y = check_X_y(X, y)
    clf = DummyClassifier(strategy="most_frequent")
    transformer = EstimatorTransformer(clone(clf))
    transformed = transformer.fit(X, y).transform(X)

    assert transformed.shape == (y.shape[0], 1)
    assert np.all(transformed == clf.fit(X, y).predict(X))


def test_set_params():
    clf = DummyClassifier(strategy="most_frequent")
    transformer = EstimatorTransformer(clf)

    transformer.set_params(estimator__strategy="stratified")
    assert clf.strategy == "stratified"


def test_get_params():
    clf = DummyClassifier(strategy="most_frequent")
    transformer = EstimatorTransformer(clf)

    assert transformer.get_params() == {
        "estimator__constant": None,
        "estimator__random_state": None,
        "estimator": clf,
        "estimator__strategy": "most_frequent",
        "predict_func": "predict",
    }


def test_shape(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    m = X.shape[0]
    pipeline = Pipeline(
        [
            (
                "ml_features",
                FeatureUnion(
                    [
                        ("model_1", EstimatorTransformer(LinearRegression())),
                        ("model_2", EstimatorTransformer(Ridge())),
                    ]
                ),
            )
        ]
    )

    assert pipeline.fit(X, y).transform(X).shape == (m, 2)


def test_shape_multitarget(random_xy_dataset_multitarget):
    X, y = random_xy_dataset_multitarget
    m = X.shape[0]
    n = y.shape[1]
    pipeline = Pipeline(
        [
            (
                "multi_ml_features",
                EstimatorTransformer(MultiOutputRegressor(Ridge()))
            )
        ]
    )
    assert pipeline.fit(X, y).transform(X).shape == (m, n)


@patch('sklego.meta.estimator_transformer.clone')
def test_kwargs(patched_clone, random_xy_dataset_clf):
    """ Test if kwargs are properly passed to an underlying estimator. """
    X, y = random_xy_dataset_clf
    estimator = Mock()
    patched_clone.return_value = estimator

    sample_weights = np.ones(shape=len(y))
    pipeline = EstimatorTransformer(estimator)
    pipeline.fit(X, y, sample_weight=sample_weights)

    # We can't use `assert_called_with` because that compares by `==` which is ambiguous
    # on numpy arrays
    np.testing.assert_array_equal(
        X, estimator.fit.call_args[0][0]
    )
    np.testing.assert_array_equal(
        y, estimator.fit.call_args[0][1]
    )
    np.testing.assert_array_equal(
        sample_weights, estimator.fit.call_args[1]['sample_weight']
    )
