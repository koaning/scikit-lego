import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from sklego.meta import SubjectiveClassifier


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
        assert 1 == pytest.approx(sum([subjective_model._posterior(pred, clazz) for pred in classes]), 0.00001)


def test_fit_stores_confusion_matrix(mocker):
    mock_inner_estimator = mocker.Mock(RandomForestClassifier)
    mock_inner_estimator.predict.return_value = np.array([42] * 90 + [23] * 10)
    mock_inner_estimator.classes_ = np.array([23, 42])
    subjective_model = SubjectiveClassifier(mock_inner_estimator, {42: 0.8, 23: 0.2})
    subjective_model.fit(np.zeros((100, 2)), np.array([42] * 80 + [23] * 20))
    assert [23, 42] == subjective_model.cfm_.index.tolist()
    assert [23, 42] == subjective_model.cfm_.columns.tolist()
    assert [[10, 10], [0, 80]] == subjective_model.cfm_.values.tolist()


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


def test_predict_proba(mocker):
    mock_inner_estimator = mocker.Mock(RandomForestClassifier)
    mock_inner_estimator.predict.return_value = np.array([0, 1])
    mock_inner_estimator.classes_ = [0, 1, 2]
    subjective_model = SubjectiveClassifier(mock_inner_estimator, {0: 0.7, 1: 0.2, 2: 0.1})
    # pretend fit() was called
    subjective_model.cfm_ = pd.DataFrame(np.array([[80, 10, 10], [10, 90, 0], [0, 0, 100]]))
    posterior_probabilities = subjective_model.predict_proba(np.zeros((2, 2)))
    assert posterior_probabilities.shape == (2, 3)
    assert np.isclose(posterior_probabilities.sum(axis=1), 1).all()


@pytest.mark.parametrize(
    'inner_estimator, prior, expected_error_msg', [
        (DBSCAN(), {'a': 1}, 'Invalid inner estimator'),
        (Ridge(), {'a': 1}, 'Invalid inner estimator'),
        (RandomForestClassifier(), {'a': 0.8, 'b': 0.1}, 'Invalid prior')
    ]
)
def test_params_failure_conditions(inner_estimator, prior, expected_error_msg):
    with pytest.raises(ValueError) as exc:
        SubjectiveClassifier(inner_estimator, prior)

    assert str(exc.value).startswith(expected_error_msg)
