import os
import numpy as np
import pandas as pd
from pkg_resources import resource_filename


def load_arrests(return_X_y=False, give_pandas=False):
    """
    Loads the arrests dataset which can serve as a benchmark for fairness. It is data on
    the police treatment of individuals arrested in Toronto for simple possession of small
    quantities of marijuana. The goal is to predict whether or not the arrestee was released
    with a summons while maintaining a degree of fairness.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)

    :Example:
    >>> from sklego.datasets import load_arrests
    >>> X, y = load_arrests(return_X_y=True)
    >>> X.shape
    (5226, 7)
    >>> y.shape
    (5226,)
    >>> load_arrests(give_pandas=True).columns
    Index(['released', 'colour', 'year', 'age', 'sex', 'employed', 'citizen',
           'checks'],
          dtype='object')

    The dataset was copied from the carData R package and can originally be found in:

    - Personal communication from Michael Friendly, York University.

    The documentation page of the dataset from the package can be viewed here:
    http://vincentarelbundock.github.io/Rdatasets/doc/carData/Arrests.html
    """
    filepath = resource_filename("sklego", os.path.join("data", "arrests.zip"))
    df = pd.read_csv(filepath)
    if give_pandas:
        return df
    X, y = (
        df[["colour", "year", "age", "sex", "employed", "citizen", "checks"]].values,
        df["released"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_chicken(return_X_y=False, give_pandas=False):
    """
    Loads the chicken dataset. The chicken data has 578 rows and 4 columns
    from an experiment on the effect of diet on early growth of chicks.
    The body weights of the chicks were measured at birth and every second
    day thereafter until day 20. They were also measured on day 21.
    There were four groups on chicks on different protein diets.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)

    :Example:

    >>> from sklego.datasets import load_chicken
    >>> X, y = load_chicken(return_X_y=True)
    >>> X.shape
    (578, 3)
    >>> y.shape
    (578,)
    >>> load_chicken(give_pandas=True).columns
    Index(['weight', 'time', 'chick', 'diet'], dtype='object')

    The datasets can be found in the following sources:

    - Crowder, M. and Hand, D. (1990), Analysis of Repeated Measures, Chapman and Hall (example 5.3)
    - Hand, D. and Crowder, M. (1996), Practical Longitudinal Data Analysis, Chapman and Hall (table A.2)
    """
    filepath = resource_filename("sklego", os.path.join("data", "chickweight.zip"))
    df = pd.read_csv(filepath)
    if give_pandas:
        return df
    if give_pandas:
        return df
    X, y = df[["time", "diet", "chick"]].values, df["weight"].values
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_abalone(return_X_y=False, give_pandas=False):
    """
    Loads the abalone dataset where the goal is to predict the gender of the creature.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)

    :Example:

    >>> from sklego.datasets import load_abalone
    >>> X, y = load_abalone(return_X_y=True)
    >>> X.shape
    (4177, 8)
    >>> y.shape
    (4177,)
    >>> load_abalone(give_pandas=True).columns
    Index(['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
           'viscera_weight', 'shell_weight', 'rings'],
          dtype='object')

    The dataset was copied from Kaggle and can originally be found in: can be found in the following sources:

    - Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994)
    "The Population Biology of Abalone (_Haliotis_ species) in Tasmania."
    Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288)
    """
    filepath = resource_filename("sklego", os.path.join("data", "abalone.zip"))
    df = pd.read_csv(filepath)
    if give_pandas:
        return df
    X = df[
        [
            "length",
            "diameter",
            "height",
            "whole_weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight",
            "rings",
        ]
    ].values
    y = df["sex"].values
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_heroes(return_X_y=False, give_pandas=False):
    """
    A dataset from a video game: "heroes of the storm". The goal of the dataset
    is to predict the attack type. Note that the pandas dataset returns more information.
    This is because we wanted to keep the X simple in the return_X_y case.
    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)

    :Example:
    >>> X, y = load_heroes(return_X_y=True)
    >>> X.shape
    (84, 2)
    >>> y.shape
    (84,)
    >>> df = load_heroes(give_pandas=True)
    >>> df.columns
    Index(['name', 'attack_type', 'role', 'health', 'attack', 'attack_spd'], dtype='object')
    """
    filepath = resource_filename("sklego", os.path.join("data", "heroes.zip"))
    df = pd.read_csv(filepath)
    if give_pandas:
        return df
    X = df[["health", "attack"]].values
    y = df["attack_type"].values
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def make_simpleseries(
    n_samples=365 * 5,
    trend=0.001,
    season_trend=0.001,
    noise=0.5,
    give_pandas=False,
    seed=None,
    stack_noise=False,
    start_date=None,
):
    """
    Generate a very simple timeseries dataset to play with. The generator
    assumes to generate daily data with a season, trend and noise.

    :param n_samples: The number of days to simulate the timeseries for.
    :param trend: The long term trend in the dataset.
    :param season_trend: The long term trend in the seasonality.
    :param noise: The noise that is applied to the dataset.
    :param give_pandas: Return a pandas dataframe instead of a numpy array.
    :param seed: The seed value for the randomness.
    :param stack_noise: Set the noise to be stacked by a cumulative sum.
    :param start_date: Also add a start date (only works if `give_pandas`=True).
    :return: numpy array unless dataframe is specified

    :Example:

    >>> from sklego.datasets import make_simpleseries
    >>> make_simpleseries(seed=42)
    array([-0.34078806, -0.61828731, -0.18458236, ..., -0.27547402,
           -0.38237413,  0.13489355])
    >>> make_simpleseries(give_pandas=True, start_date="2018-01-01", seed=42).head(3)
             yt       date
    0 -0.340788 2018-01-01
    1 -0.618287 2018-01-02
    2 -0.184582 2018-01-03
    """
    if seed:
        np.random.seed(seed)
    time = np.arange(0, n_samples)
    noise = np.random.normal(0, noise, n_samples)
    if stack_noise:
        noise = noise.cumsum()
    r1, r2 = np.random.normal(0, 1, 2)
    seasonality = r1 * np.sin(time / 365 * 2 * np.pi) + r2 * np.cos(
        time / 365 * 4 * np.pi + 1
    )
    result = seasonality + season_trend * seasonality * time + trend * time + noise
    if give_pandas:
        if start_date:
            stamps = pd.date_range(start_date, periods=n_samples)
            return pd.DataFrame({"yt": result, "date": stamps})
        return pd.DataFrame({"yt": result})
    return result
