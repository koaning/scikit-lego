import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from sklego.meta import SubjectiveClassifier


@pytest.mark.parametrize(
    'classes, prior, cfm, first_class_posterior', [
        ([0, 1], [0.8, 0.2], [[90, 10], [10, 90]], 0.973),  # numeric classes
        (['a', 'b'], [0.8, 0.2], [[90, 10], [10, 90]], 0.973),  # char classes
        ([False, True], [0.8, 0.2], [[90, 10], [10, 90]], 0.973),  # bool classes
        (['a', 'b', 'c'], [0.8, 0.1, 0.1], [[80, 10, 10], [10, 90, 0], [0, 0, 100]], 0.985),  # n classes
        ([0, 1], [0.2, 0.8], [[0, 100], [0, 100]], 0)  # failure to predict class by inner estimator
    ]
)
def test_posterior_computation(classes, prior, cfm, first_class_posterior):
    subjective_model = SubjectiveClassifier(RandomForestClassifier(), dict(zip(classes, prior)))
    # pretend fit() was called
    subjective_model.estimator.classes_ = np.array(classes)
    subjective_model.cfm_ = pd.DataFrame(np.array(cfm), index=classes, columns=classes)
    assert first_class_posterior == pytest.approx(subjective_model._posterior(classes[0], classes[0]), 0.001)
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
