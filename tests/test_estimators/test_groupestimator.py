import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklego.group_estimator import GroupEstimator
from sklego.dummy import RandomRegressor
from tests.conftest import id_func



@pytest.fixture
def simple_dataframe():
    return pd.DataFrame.from_dict(
        {
            'col_1': list(np.concatenate((np.zeros(5), np.ones(5)))),
            'col_2': list(np.random.normal(1,2,(10,1))),
            'Class': list(np.random.normal(1,2,(10,1)))   
        }
    )


def test_estimator(simple_dataframe):
    # test grouping
    x, y = GroupEstimator(DummyClassifier(), "col_1").separate_dataset(simple_dataframe, simple_dataframe["Class"], "col_1")
    assert len(x) == 2 and len(y) == 2

    # test grouping with model
    res = GroupEstimator(DummyClassifier(), "col_1").fit(simple_dataframe, simple_dataframe["Class"]).predict(simple_dataframe)
    assert np.array(res).shape == (2, simple_dataframe.shape[0])


@pytest.mark.parametrize("estimator", [
    RandomRegressor(),
], ids=id_func)
def test_shape_regression(estimator, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    X_len = len(X[0])
    X = pd.DataFrame(X, 
                    columns=[str(i) for i in range(X_len)]
                    )
    len_class = X['0'].unique().shape[0]
    y = pd.DataFrame(y)[0]
    assert len(GroupEstimator(estimator, "0").fit(X, y).predict(X)) == len_class
