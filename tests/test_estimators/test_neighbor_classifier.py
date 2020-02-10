import numpy as np

import pytest

from sklego.neighbors import BayesianKernelDensityClassifier
from sklego.common import flatten
from sklego.testing import check_shape_remains_same_classifier
from tests.conftest import nonmeta_checks, general_checks, estimator_checks


@pytest.mark.parametrize(
    "test_fn",
    flatten(
        [
            nonmeta_checks,
            general_checks,
            estimator_checks.check_classifier_data_not_an_array,
            estimator_checks.check_classifiers_one_label,
            estimator_checks.check_classifiers_classes,
            estimator_checks.check_classifiers_train,
            estimator_checks.check_supervised_y_2d,
            estimator_checks.check_supervised_y_no_nan,
            estimator_checks.check_estimators_unfitted,
            check_shape_remains_same_classifier,
        ]
    ),
)
def test_estimator_checks(test_fn):
    test_fn(BayesianKernelDensityClassifier.__name__, BayesianKernelDensityClassifier())


def test_trivial_classification():
    # Two linearly separable mvn should have a 100% prediction accuracy
    x = np.concatenate(
        [np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))]
    )
    y = np.concatenate([np.zeros(100), np.ones(100)])

    model = BayesianKernelDensityClassifier().fit(x, y)
    assert (model.predict(x) == y).all()


def test_n_jobs_param():
    # Dummy data
    x = np.concatenate(
        [np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))]
    )
    y = np.concatenate([np.zeros(100), np.ones(100)])

    BayesianKernelDensityClassifier()  # Default params
    BayesianKernelDensityClassifier(n_jobs=None).fit(x, y)  # None specified
    BayesianKernelDensityClassifier(n_jobs=2).fit(x, y)  # 2 Jobs

    with pytest.raises(ValueError):
        BayesianKernelDensityClassifier(n_jobs=1.23)  # No floats
        BayesianKernelDensityClassifier(n_jobs=1.23).fit(x, y)  # No floats
        BayesianKernelDensityClassifier(n_jobs=0).fit(x, y)  # No Zero
