import pytest
import numpy as np

from sklearn.linear_model import LinearRegression
from sklego.meta import GroupedEstimator
from sklego.datasets import load_chicken


def test_chickweight_np():
    df = load_chicken()
    mod = GroupedEstimator(estimator=LinearRegression(), groupby="diet")
    X, y = df[['time', 'diet']].values.reshape(-1, 1), df['weight'].values
    mod.fit(X, y)


def test_chickweight_df():
    df = load_chicken()
    mod = GroupedEstimator(estimator=LinearRegression(), groupby="diet")
    mod.fit(df[['time', 'diet']], df['weight'])
