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
    subjective_model = SubjectiveClassifier(RandomForestClassifier, dict(zip(classes, prior)))
    # pretend fit() was called
    subjective_model.estimator.classes_ = np.array(classes)
    subjective_model.cfm_ = pd.DataFrame(np.array(cfm), index=classes, columns=classes)
    assert first_class_posterior == pytest.approx(subjective_model._posterior(classes[0], classes[0]), 0.001)
    for clazz in classes:
        assert 1 == pytest.approx(sum([subjective_model._posterior(pred, clazz) for pred in classes]), 0.00001)
