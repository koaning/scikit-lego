import pytest
import pandas as pd
from sklearn.utils import estimator_checks
from sklearn.linear_model import LinearRegression

from sklego.common import flatten
from sklego.meta import GroupedEstimator
from sklego.datasets import load_chicken


@pytest.mark.parametrize("test_fn", flatten([
    estimator_checks.check_fit_score_takes_y,
    estimator_checks.check_sample_weights_invariance,
    estimator_checks.check_estimators_empty_data_messages,
    estimator_checks.check_estimators_nan_inf,
    estimator_checks.check_estimators_overwrite_params,
    estimator_checks.check_estimators_pickle,
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit1d,
    estimator_checks.check_dont_overwrite_parameters,
    estimator_checks.check_sample_weights_invariance,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_sample_weights_list,
    estimator_checks.check_sample_weights_pandas_series,
    estimator_checks.check_set_params,
]))
def test_estimator_checks(test_fn):
    clf = GroupedEstimator(estimator=LinearRegression(), groups=[0], use_fallback=True)
    test_fn(GroupedEstimator.__name__ + "_fallback", clf)

    clf = GroupedEstimator(estimator=LinearRegression(), groups=[0], use_fallback=False)
    test_fn(GroupedEstimator.__name__ + "_nofallback", clf)


def test_chickweight_df1_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}


def test_chickweight_df2_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="chick")
    mod.fit(df[['time', 'chick']], df['weight'])
    assert set(mod.estimators_.keys()) == set(range(1, 50 + 1))


def test_chickweight_can_do_fallback():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    assert mod.predict(to_predict).shape == (2,)
    assert mod.predict(to_predict)[0] == mod.predict(to_predict)[1]


def test_fallback_can_raise_error():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(),
                           groups="diet",
                           use_fallback=False)
    mod.fit(df[['time', 'diet']], df['weight'])
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    with pytest.raises(ValueError):
        mod.predict(to_predict)


def test_chickweight_raise_error_cols_missing1():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
    with pytest.raises(ValueError):
        mod.predict(df[['time', 'chick']])


def test_chickweight_raise_error_cols_missing2():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
    with pytest.raises(ValueError):
        mod.predict(df[['diet', 'chick']])


def test_chickweight_np_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups=[1, 2])
    mod.fit(df[['time', 'chick', 'diet']].values, df['weight'].values)
    # there should still only be 50 groups on this dataset
    assert len(mod.estimators_.keys()) == 50
