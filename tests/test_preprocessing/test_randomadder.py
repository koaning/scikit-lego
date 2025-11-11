import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.preprocessing import RandomAdder


@parametrize_with_checks([RandomAdder()])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_transformer_data_not_an_array",  # hash only supports a few types
        "check_pipeline_consistency",
        "check_transformer_general",
    }:
        pytest.skip("RandomAdder is a TrainOnlyTransformer")

    check(estimator)


def test_dtype_regression(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert RandomAdder().fit(X, y).transform(X).dtype == float


def test_dtype_classification(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert RandomAdder().fit(X, y).transform(X).dtype == float


def test_only_transform_train(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_adder = RandomAdder()
    random_adder.fit(X_train, y_train)

    assert np.all(random_adder.transform(X_train) != X_train)
    assert np.all(random_adder.transform(X_test) == X_test)
