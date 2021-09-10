import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from sklego.common import flatten
from sklego.preprocessing import RandomAdder


from tests.conftest import select_tests, transformer_checks, nonmeta_checks, general_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, transformer_checks, nonmeta_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_methods_subset_invariance",
            "check_transformer_data_not_an_array",
            "check_transformer_general",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
def test_estimator_checks(test_fn):
    adder = RandomAdder()
    test_fn(RandomAdder.__name__, adder)


def test_dtype_regression(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert RandomAdder().fit(X, y).transform(X).dtype == np.float


def test_dtype_classification(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert RandomAdder().fit(X, y).transform(X).dtype == np.float


def test_only_transform_train(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_adder = RandomAdder()
    random_adder.fit(X_train, y_train)

    assert np.all(random_adder.transform(X_train) != X_train)
    assert np.all(random_adder.transform(X_test) == X_test)
