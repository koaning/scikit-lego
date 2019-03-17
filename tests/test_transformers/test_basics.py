import pytest

from sklego.transformers import RandomAdder
from tests.conftest import id_func


@pytest.mark.parametrize("transformer", [
    RandomAdder(),
], ids=id_func)
def test_same_shape_regression(transformer, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert transformer.fit(X, y).transform(X).shape == X.shape


@pytest.mark.parametrize("transformer", [
    RandomAdder(),
], ids=id_func)
def test_shape_classification(transformer, random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert transformer.fit(X, y).transform(X).shape == X.shape
