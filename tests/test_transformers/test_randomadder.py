import numpy as np
from sklearn.model_selection import train_test_split

from sklego.transformers import RandomAdder


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
