import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklego.meta import GroupedEstimator
from sklego.datasets import load_chicken
from tests.conftest import nonmeta_checks, general_checks, classifier_checks


@pytest.mark.parametrize("test_fn", flatten([
    nonmeta_checks,
    general_checks,
    classifier_checks,
    check_shape_remains_same_classifier
]))
def test_estimator_checks(test_fn):
    clf = GroupedEstimator()
    test_fn(GroupedEstimator.__name__, clf)

def test_chickweight_df1_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groupby="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}


def test_chickweight_df2_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groupby="chick")
    mod.fit(df[['time', 'chick']], df['weight'])
    assert set(mod.estimators_.keys()) == set(range(1, 50 + 1))


def test_chickweight_can_do_fallback():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groupby="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    assert mod.predict(to_predict).shape == (2,)
    assert mod.predict(to_predict)[0] == mod.predict(to_predict)[1]


def test_fallback_can_raise_error():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(),
                           groupby="diet",
                           use_fallback=False)
    mod.fit(df[['time', 'diet']], df['weight'])
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    with pytest.raises(ValueError):
        mod.predict(to_predict)


def test_chickweight_np_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groupby=[1, 2])
    mod.fit(df[['time', 'chick', 'diet']].values, df['weight'].values)
    # there should still only be 50 groups on this dataset
    assert len(mod.estimators_.keys()) == 50
