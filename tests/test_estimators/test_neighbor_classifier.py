import numpy as np

import pytest

from sklego.neighbors import BayesianKernelDensityClassifier
from sklego.common import flatten
from sklego.testing import check_shape_remains_same_classifier
from tests.conftest import nonmeta_checks, general_checks, estimator_checks


@pytest.fixture()
def simple_dataset():
    # Two linearly separable mvn should have a 100% prediction accuracy
    x = np.concatenate(
        [np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))]
    )
    y = np.concatenate([np.zeros(100), np.ones(100)])
    return x, y


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


def test_trivial_classification(simple_dataset):
    x, y = simple_dataset
    model = BayesianKernelDensityClassifier().fit(x, y)
    assert (model.predict(x) == y).all()


@pytest.mark.parametrize("n_jobs", [None, -1, 2, 1])
def test_n_jobs_passes(simple_dataset, n_jobs):
    x, y = simple_dataset
    BayesianKernelDensityClassifier(n_jobs=n_jobs).fit(x, y).score(x, y)


@pytest.mark.parametrize("n_jobs", [0, 1.23])
def test_n_jobs_params_fails(simple_dataset, n_jobs):
    x, y = simple_dataset
    with pytest.raises(ValueError):
        BayesianKernelDensityClassifier(n_jobs=n_jobs).fit(x, y).score(x, y)
