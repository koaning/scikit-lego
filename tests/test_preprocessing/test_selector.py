
import pytest
import pandas as pd
from sklego.preprocessing import PandasTypeSelector, ColumnSelector
from tests.conftest import id_func


@pytest.mark.parametrize("transformer", [
    PandasTypeSelector(include=['number']),
    ColumnSelector([0])
], ids=id_func)
def test_len_regression(transformer, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    X = pd.DataFrame(X)
    assert transformer.fit(X, y).transform(X).shape[0] == X.shape[0]


@pytest.mark.parametrize("transformer", [
    PandasTypeSelector(include=['number']),
    ColumnSelector([0])
], ids=id_func)
def test_len_classification(transformer, random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X = pd.DataFrame(X)
    assert transformer.fit(X, y).transform(X).shape[0] == X.shape[0]
