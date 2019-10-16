import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression

from sklego.common import flatten
from sklego.meta import SubjectiveClassifier
from tests.conftest import general_checks, classifier_checks


@pytest.mark.parametrize("test_fn", flatten([
    general_checks,
    classifier_checks
]))
def test_estimator_checks_classification(test_fn):
    if test_fn.__name__ == 'check_classifiers_classes':
        prior = {'one': 0.1, 'two': 0.1, 'three': 0.1, -1: 0.1, 1: 0.6}  # nonsensical prior to make sklearn check pass
    else:
        prior = {0: 0.7, 1: 0.2, 2: 0.1}

    # Some of the sklearn checkers generate random y data with 3 classes, so prior needs to have these classes
    estimator = SubjectiveClassifier(LogisticRegression(), prior)
    test_fn(SubjectiveClassifier.__name__, estimator)


@pytest.mark.parametrize(
    'classes, prior, cfm, first_class_posterior', [
        ([0, 1], [0.8, 0.2], [[90, 10], [10, 90]], 0.973),  # numeric classes
        (['a', 'b'], [0.8, 0.2], [[90, 10], [10, 90]], 0.973),  # char classes
        ([False, True], [0.8, 0.2], [[90, 10], [10, 90]], 0.973),  # bool classes
        (['a', 'b', 'c'], [0.8, 0.1, 0.1], [[80, 10, 10], [10, 90, 0], [0, 0, 100]], 0.985),  # n classes
        ([0, 1], [0.8, 0.2], [[100, 0], [0, 100]], 1.0),  # "perfect" confusion matrix (no FP) -> prior is ignored
        ([0, 1], [0.2, 0.8], [[0, 100], [0, 100]], 0.2),  # failure to predict class by inner estimator
        ([0, 1, 2], [0.1, 0.1, 0.8], [[0, 0, 100], [0, 0, 100], [0, 0, 100]], 0.1),  # extremely biased, n classes
        ([0, 1, 2], [0.2, 0.1, 0.7], [[80, 0, 20], [0, 0, 100], [10, 0, 90]], 0.696)  # biased, n classes
    ]
)
def test_posterior_computation(classes, prior, cfm, first_class_posterior):
    subjective_model = SubjectiveClassifier(RandomForestClassifier(), dict(zip(classes, prior)))
    # pretend fit() was called
    subjective_model.estimator.classes_ = np.array(classes)
    subjective_model.cfm_ = pd.DataFrame(np.array(cfm), index=classes, columns=classes)
    assert pytest.approx(subjective_model._posterior(classes[0], classes[0]), 0.001) == first_class_posterior
    for clazz in classes:
        assert pytest.approx(sum([subjective_model._posterior(pred, clazz) for pred in classes]), 0.00001) == 1


def test_fit_stores_confusion_matrix(mocker):
    mock_inner_estimator = mocker.Mock(RandomForestClassifier)
    mock_inner_estimator.predict.return_value = np.array([42] * 90 + [23] * 10)
    mock_inner_estimator.classes_ = np.array([23, 42])
    subjective_model = SubjectiveClassifier(mock_inner_estimator, {42: 0.8, 23: 0.2})
    subjective_model.fit(np.zeros((100, 2)), np.array([42] * 80 + [23] * 20))
    assert subjective_model.cfm_.index.tolist() == [23, 42]
    assert subjective_model.cfm_.columns.tolist() == [23, 42]
    assert subjective_model.cfm_.values.tolist() == [[10, 10], [0, 80]]


@pytest.mark.parametrize(
    'prior, y', [
        ({'a': 0.8, 'b': 0.2}, ['a', 'c']),  # class from train data not defined in prior
        ({'a': 0.8, 'b': 0.2}, [0, 1]),  # different data types
    ]
)
def test_fit_y_data_inconsistent_with_prior_failure_conditions(prior, y):
    with pytest.raises(ValueError) as exc:
        SubjectiveClassifier(RandomForestClassifier(), prior).fit(np.zeros((len(y), 2)), np.array(y))

    assert str(exc.value).startswith('Training data is inconsistent with prior')


@pytest.mark.parametrize(
    'weights,y_hats,expected_probas', [
        ([0.8, 0.2], [[1, 0], [0.5, 0.5], [0.8, 0.2]], [[1, 0], [0.8, 0.2], [0.94, 0.06]]),
        ([0.5, 0.5], [[1, 0], [0.5, 0.5], [0.8, 0.2]], [[1, 0], [0.5, 0.5], [0.8, 0.2]]),
        ([[0.8, 0.2], [0.5, 0.5]], [[1, 0], [0.8, 0.2]], [[1, 0], [0.8, 0.2]])
    ]
)
def test_weighted_proba(weights, y_hats, expected_probas):
    assert np.isclose(
        SubjectiveClassifier._weighted_proba(np.array(weights), np.array(y_hats)), np.array(expected_probas), atol=1e-02
    ).all()


@pytest.mark.parametrize(
    'evidence_type,expected_probas', [
        ('predict_proba', [[0.94, 0.06], [1, 0], [0.8, 0.2], [0.5, 0.5]]),
        ('confusion_matrix', [[0.97, 0.03], [0.97, 0.03], [0.97, 0.03], [0.47, 0.53]]),
        ('both', [[0.99, 0.01], [1, 0], [0.97, 0.03], [0.18, 0.82]])
    ]
)
def test_predict_proba(mocker, evidence_type, expected_probas):
    mock_inner_estimator = mocker.Mock(RandomForestClassifier)
    mock_inner_estimator.predict_proba.return_value = np.array([[0.8, 0.2], [1, 0], [0.5, 0.5], [0.2, 0.8]])
    mock_inner_estimator.classes_ = np.array([0, 1])
    subjective_model = SubjectiveClassifier(mock_inner_estimator, {0: 0.8, 1: 0.2}, evidence=evidence_type)
    # pretend fit() was called
    subjective_model.cfm_ = pd.DataFrame(np.array([[80, 20], [10, 90]]))
    posterior_probabilities = subjective_model.predict_proba(np.zeros((4, 2)))
    assert posterior_probabilities.shape == (4, 2)
    assert np.isclose(posterior_probabilities, np.array(expected_probas), atol=0.01).all()


@pytest.mark.parametrize(
    'inner_estimator, prior, evidence, expected_error_msg', [
        (DBSCAN(), {'a': 1}, 'both', 'Invalid inner estimator'),
        (Ridge(), {'a': 1}, 'predict_proba', 'Invalid inner estimator'),
        (RandomForestClassifier(), {'a': 0.8, 'b': 0.1}, 'confusion_matrix', 'Invalid prior'),
        (RandomForestClassifier(), {'a': 0.8, 'b': 0.2}, 'foo_evidence', 'Invalid evidence')
    ]
)
def test_params_failure_conditions(inner_estimator, prior, evidence, expected_error_msg):
    with pytest.raises(ValueError) as exc:
        SubjectiveClassifier(inner_estimator, prior, evidence)

    assert str(exc.value).startswith(expected_error_msg)
