from sklearn.linear_model import LinearRegression
from sklego.meta import GroupedEstimator
from sklego.datasets import load_chicken


def test_chickweight_df():
    df = load_chicken()
    mod = GroupedEstimator(estimator=LinearRegression(), groupby="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
