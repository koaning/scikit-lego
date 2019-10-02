import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from sklego.meta import SubjectiveClassifier


def test_posterior_computation():
    subjective_model = SubjectiveClassifier(RandomForestClassifier, {0: 0.8, 1: 0.2})
    # pretend fit() was called
    subjective_model.estimator.classes_ = np.array([0, 1])
    subjective_model.cfm_ = pd.DataFrame(np.array([[90, 10], [10, 90]]), index=[0, 1], columns=[0, 1])
    assert 1 == pytest.approx(subjective_model._posterior(0, 0) + subjective_model._posterior(1, 0), 0.00001)
    assert 1 == pytest.approx(subjective_model._posterior(1, 1) + subjective_model._posterior(0, 1), 0.00001)
