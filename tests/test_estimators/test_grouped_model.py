from sklearn.linear_model import LinearRegression
from sklego.meta import GroupedEstimator
from sklego.datasets import load_chicken


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


def test_chickweight_np_keys():
    df = load_chicken(give_pandas=True)
    mod = GroupedEstimator(estimator=LinearRegression(), groupby=[1, 2])
    mod.fit(df[['time', 'chick', 'diet']].values, df['weight'].values)
    # there should still only be 50 groups on this dataset
    assert len(mod.estimators_.keys()) == 50