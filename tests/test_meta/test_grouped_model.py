import pytest
import pandas as pd
import numpy as np
from sklearn.utils import estimator_checks
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

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
    estimator_checks.check_fit2d_1sample,
    # estimator_checks.check_fit1d not tested because in 1d we cannot have both groups and data
    estimator_checks.check_dont_overwrite_parameters,
    estimator_checks.check_sample_weights_invariance,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_sample_weights_list,
    estimator_checks.check_sample_weights_pandas_series,
    estimator_checks.check_set_params,
]))
def test_estimator_checks(test_fn):
    clf = GroupedEstimator(estimator=LinearRegression(), groups=[0], use_fallback=True, shrinkage=None)
    test_fn(GroupedEstimator.__name__ + "_fallback", clf)

    clf = GroupedEstimator(estimator=LinearRegression(), groups=[0], use_fallback=False, shrinkage=None)
    test_fn(GroupedEstimator.__name__ + "_nofallback", clf)


def test_chickweight_df1_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet", shrinkage=None)
    mod.fit(df[['time', 'diet']], df['weight'])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}


def test_chickweight_df2_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="chick", shrinkage=None)
    mod.fit(df[['time', 'chick']], df['weight'])
    assert set(mod.estimators_.keys()) == set(range(1, 50 + 1))


def test_chickweight_can_do_fallback():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet", shrinkage=None)
    mod.fit(df[['time', 'diet']], df['weight'])
    assert set(mod.estimators_.keys()) == {1, 2, 3, 4}
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    assert mod.predict(to_predict).shape == (2,)
    assert mod.predict(to_predict)[0] == mod.predict(to_predict)[1]


def test_fallback_can_raise_error():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(),
                           groups="diet",
                           use_fallback=False,
                           shrinkage=None)
    mod.fit(df[['time', 'diet']], df['weight'])
    to_predict = pd.DataFrame({"time": [21, 21], "diet": [5, 6]})
    with pytest.raises(ValueError):
        mod.predict(to_predict)


def test_chickweight_raise_error_cols_missing1():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet", shrinkage=None)
    mod.fit(df[['time', 'diet']], df['weight'])
    with pytest.raises(ValueError):
        mod.predict(df[['time', 'chick']])


def test_chickweight_raise_error_cols_missing2():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups="diet", shrinkage=None)
    mod.fit(df[['time', 'diet']], df['weight'])
    with pytest.raises(ValueError):
        mod.predict(df[['diet', 'chick']])


def test_chickweight_np_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groups=[1, 2], shrinkage=None)
    mod.fit(df[['time', 'chick', 'diet']].values, df['weight'].values)
    # there should still only be 50 groups on this dataset
    assert len(mod.estimators_.keys()) == 50


def test_chickweigt_string_groups():

    df = load_chicken(give_pandas=True)
    df['diet'] = ['omgomgomg' + s for s in df['diet'].astype(str)]

    X = df[['time', 'diet']]
    X_np = np.array(X)

    y = df['weight']

    # This should NOT raise errors
    GroupedEstimator(LinearRegression(), groups=['diet'], shrinkage=None).fit(X, y).predict(X)
    GroupedEstimator(LinearRegression(), groups=1, shrinkage=None).fit(X_np, y).predict(X_np)


@pytest.fixture
def shrinkage_data():
    df = pd.DataFrame({
        "Planet": ["Earth", "Earth", "Earth", "Earth"],
        "Country": ["NL", "NL", "BE", "BE"],
        "City": ["Amsterdam", "Rotterdam", "Antwerp", "Brussels"],
        "Target": [1, 3, 2, 4]
    })

    means = {"Earth": 2.5, "NL": 2, "BE": 3, "Amsterdam": 1, "Rotterdam": 3, "Antwerp": 2, "Brussels": 4}

    return df, means


def test_constant_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df['Target']

    shrink_est = GroupedEstimator(
        DummyRegressor(), ["Planet", 'Country', 'City'], shrinkage="constant",
        alpha=0.1
    )

    shrinkage_factors = np.array([0.01, 0.09, 0.9])

    shrink_est.fit(X, y)

    expected_prediction = [
        np.array([means["Earth"], means["NL"], means["Amsterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["NL"], means["Rotterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Antwerp"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Brussels"]]) @ shrinkage_factors,
    ]

    assert expected_prediction == shrink_est.predict(X).tolist()


def test_relative_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df['Target']

    shrink_est = GroupedEstimator(
        DummyRegressor(), ["Planet", 'Country', 'City'], shrinkage="relative",
    )

    shrinkage_factors = np.array([4, 2, 1]) / 7

    shrink_est.fit(X, y)

    expected_prediction = [
        np.array([means["Earth"], means["NL"], means["Amsterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["NL"], means["Rotterdam"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Antwerp"]]) @ shrinkage_factors,
        np.array([means["Earth"], means["BE"], means["Brussels"]]) @ shrinkage_factors,
    ]

    assert expected_prediction == shrink_est.predict(X).tolist()


def test_min_n_obs_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df['Target']

    shrink_est = GroupedEstimator(
        DummyRegressor(), ["Planet", 'Country', 'City'], shrinkage="min_n_obs",
        min_n_obs=2
    )

    shrink_est.fit(X, y)

    expected_prediction = [
        means["NL"],
        means["NL"],
        means["BE"],
        means["BE"],
    ]

    assert expected_prediction == shrink_est.predict(X).tolist()


def test_unexisting_shrinkage_func(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df['Target']

    with pytest.raises(ValueError):
        unexisting_func = "some_highly_unlikely_function_name"

        shrink_est = GroupedEstimator(
            DummyRegressor(), ["Planet", 'Country'], shrinkage=unexisting_func,
            min_n_obs=2
        )

        shrink_est.fit(X, y)


def test_unseen_groups_shrinkage(shrinkage_data):
    df, means = shrinkage_data

    X, y = df.drop(columns="Target"), df['Target']

    shrink_est = GroupedEstimator(
        DummyRegressor(), ["Planet", 'Country', 'City'], shrinkage="constant",
        alpha=0.1
    )

    shrink_est.fit(X, y)

    unseen_group = pd.DataFrame({"Planet": ["Earth"], 'Country': ["DE"], 'City': ["Hamburg"]})

    with pytest.raises(ValueError):
        shrink_est.predict(X=pd.concat([unseen_group] * 4, axis=0))


def test_bad_shrinkage_value_error():
    with pytest.raises(ValueError):
        df = load_chicken(give_pandas=True)
        mod = GroupedEstimator(estimator=LinearRegression(), groups="diet", shrinkage="dinosaurhead")
        mod.fit(df[['time', 'diet']], df['weight'])
