import sys

import numpy as np
import pandas as pd

if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources  # pragma: no cover
else:
    import importlib_resources as importlib_resources  # pragma: no cover


from sklearn.datasets import fetch_openml


def load_penguins(return_X_y=False, as_frame=False):
    """Loads the penguins dataset, which is a lovely alternative for the iris dataset.
    We've added this dataset for educational use.

    Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of
    the Long Term Ecological Research Network. The goal of the dataset is to predict which species of penguin a penguin
    belongs to.

    This data originally appeared as a R package and R users can find this data in the
    [palmerpenguins package](https://github.com/allisonhorst/palmerpenguins).
    You can also go to the repository for some lovely images that explain the dataset.

    To cite this dataset in publications use:

        Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism
        and Environmental Variability within a Community of Antarctic
        Penguins (Genus Pygoscelis). PLoS ONE 9(3): e90081.
        https://doi.org/10.1371/journal.pone.0090081

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    -------
    ```py
    from sklego.datasets import load_penguins
    X, y = load_penguins(return_X_y=True)

    X.shape
    # (344, 6)
    y.shape
    # (344,)

    load_penguins(as_frame=True).columns
    # Index(['species', 'island', 'bill_length_mm', 'bill_depth_mm',
    #    'flipper_length_mm', 'body_mass_g', 'sex'],
    #   dtype='object')
    ```

    Notes
    -----
    Anyone interested in publishing the data should contact
    [`Dr. Kristen Gorman`](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)
    about analysis and working together on any final products.

    !!! quote "From Gorman et al. (2014)"

        Data reported here are publicly available within the
        [PAL-LTER data system (datasets 219, 220, and 221)](http://oceaninformatics.ucsd.edu/datazoo/data/pallter/datasets).

        Individuals interested in using these data are therefore expected to follow the [US LTER Network's Data Access
        Policy, Requirements and Use Agreement](https://lternet.edu/data-access-policy/)

    Please cite data using the following
    ------------------------------------
    **Adélie penguins:**

      - [Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size measurements and isotopic signatures of
        foraging among adult male and female Adélie penguins (*Pygoscelis adeliae*) nesting along the Palmer Archipelago
        near Palmer Station, 2007-2009 ver 5. Environmental Data
        Initiative](https://doi.org/10.6073/pasta/98b16d7d563f265cb52372c8ca99e60f).

        (Accessed 2020-06-08).

    **Gentoo penguins:**

      - [Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size measurements and isotopic signatures of
        foraging among adult male and female Gentoo penguin (*Pygoscelis papua*) nesting along the Palmer Archipelago
        near Palmer Station, 2007-2009 ver 5. Environmental Data
        Initiative](https://doi.org/10.6073/pasta/7fca67fb28d56ee2ffa3d9370ebda689).

        (Accessed 2020-06-08).

    **Chinstrap penguins:**

      - [Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size measurements and isotopic signatures of
        foraging among adult male and female Chinstrap penguin (*Pygoscelis antarcticus*) nesting along the Palmer
        Archipelago near Palmer Station, 2007-2009 ver 6.
        Environmental Data Initiative](https://doi.org/10.6073/pasta/c14dfcfada8ea13a17536e73eb6fbe9e).

        (Accessed 2020-06-08).
    """
    filepath = importlib_resources.files("sklego") / "data" / "penguins.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[
            [
                "island",
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "sex",
            ]
        ].to_numpy(),
        df["species"].to_numpy(),
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_arrests(return_X_y=False, as_frame=False):
    """Loads the arrests dataset which can serve as a benchmark for fairness.

    It is data on the police treatment of individuals arrested in Toronto for simple possession of small quantities of
    marijuana. The goal is to predict whether or not the arrestee was released with a summons while maintaining a
    degree of fairness.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    -------
    ```py
    from sklego.datasets import load_arrests
    X, y = load_arrests(return_X_y=True)

    X.shape
    # (5226, 7)
    y.shape
    # (5226,)

    load_arrests(as_frame=True).columns
    # Index(['released', 'colour', 'year', 'age', 'sex', 'employed', 'citizen',
    #   'checks'],
    #  dtype='object')
    ```

    The dataset was copied from the carData R package
    ([dataset documentation](https://vincentarelbundock.github.io/Rdatasets/doc/carData/Arrests.html))
    and can originally be found in:

    - Personal communication from Michael Friendly, York University.
    """
    filepath = importlib_resources.files("sklego") / "data" / "arrests.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[["colour", "year", "age", "sex", "employed", "citizen", "checks"]].to_numpy(),
        df["released"].to_numpy(),
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_chicken(return_X_y=False, as_frame=False):
    """Loads the chicken dataset.

    The chicken data has 578 rows and 4 columns from an experiment on the effect of diet on early growth of chicks.
    The body weights of the chicks were measured at birth and every second day thereafter until day 20.
    They were also measured on day 21. There were four groups on chicks on different protein diets.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    -------
    ```py
    from sklego.datasets import load_chicken
    X, y = load_chicken(return_X_y=True)

    X.shape
    # (578, 3)
    y.shape
    # (578,)

    load_chicken(as_frame=True).columns
    # Index(['weight', 'time', 'chick', 'diet'], dtype='object')
    ```

    The datasets can be found in the following sources:

    - Crowder, M. and Hand, D. (1990), Analysis of Repeated Measures, Chapman and Hall (example 5.3)
    - Hand, D. and Crowder, M. (1996), Practical Longitudinal Data Analysis, Chapman and Hall (table A.2)
    """
    filepath = importlib_resources.files("sklego") / "data" / "chickweight.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = df[["time", "diet", "chick"]].to_numpy(), df["weight"].to_numpy()
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_abalone(return_X_y=False, as_frame=False):
    """
    Loads the abalone dataset where the goal is to predict the gender of the creature.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    ---------
    ```py
    from sklego.datasets import load_abalone
    X, y = load_abalone(return_X_y=True)

    X.shape
    # (4177, 8)
    y.shape
    # (4177,)

    load_abalone(as_frame=True).columns
    # Index(['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
    #        'viscera_weight', 'shell_weight', 'rings'],
    #       dtype='object')
    ```

    The dataset was copied from Kaggle and can originally be found in the following sources:

    - Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994)

        "The Population Biology of Abalone (_Haliotis_ species) in Tasmania."

        Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288)
    """
    filepath = importlib_resources.files("sklego") / "data" / "abalone.zip"
    df = pd.read_csv(filepath)
    if as_frame:
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
    ].to_numpy()
    y = df["sex"].to_numpy()
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_heroes(return_X_y=False, as_frame=False):
    """A dataset from the video game [Heroes of the storm](https://heroesofthestorm.blizzard.com/en-us/).

    The goal of the dataset is to predict the attack type. Note that the pandas dataset returns more information.
    This is because we wanted to keep the X simple in the return_X_y case.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    --------
    ```py
    from sklego.datasets import load_heroes
    X, y = load_heroes(return_X_y=True)

    X.shape
    # (84, 2)
    y.shape
    # (84,)

    load_heroes(as_frame=True).columns
    # Index(['name', 'attack_type', 'role', 'health', 'attack', 'attack_spd'], dtype='object')
    ```
    """
    filepath = importlib_resources.files("sklego") / "data" / "heroes.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X = df[["health", "attack"]].to_numpy()
    y = df["attack_type"].to_numpy()
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_hearts(return_X_y=False, as_frame=False):
    """Loads the Cleveland Heart Diseases dataset.

    The goal is to predict the presence of a heart disease (target values 1, 2, 3, and 4).
    The data originates from research to heart diseases by four institutions and originally contains 76 attributes.
    Yet, all published experiments refer to using a subset of 13 features and one target.
    This implementation loads the Cleveland dataset of the research which is the only set used by ML researchers
    to this date.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a dict object.
    as_frame : bool, default=False
        If True, returns `data` as a pandas DataFrame  instead of X, y matrices.

    Examples
    --------
    ```py
    from sklego.datasets import load_hearts
    X, y = load_hearts(return_X_y=True)

    X.shape
    # (303, 13)
    y.shape
    # (303,)

    load_hearts(as_frame=True).columns
    # Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    #        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
    #       dtype='object')
    ```

    The dataset can originally be found in the
    [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

    The responsible institutions for the entire database are:

    1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

    The documentation of the dataset can be viewed at:
    https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
    """
    filepath = importlib_resources.files("sklego") / "data" / "hearts.zip"
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X = df[
        [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
    ].to_numpy()
    y = df["target"].to_numpy()
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def make_simpleseries(
    n_samples=365 * 5,
    trend=0.001,
    season_trend=0.001,
    noise=0.5,
    as_frame=False,
    seed=None,
    stack_noise=False,
    start_date=None,
):
    """Generate a very simple timeseries dataset to play with.

    The generator returns a daily time-series with a yearly seasonality, trend, and noise.

    Parameters
    ----------
    n_samples : int, default=365 * 5
        The number of days to simulate the timeseries for.
    trend : float, default=0.001
        The long term trend in the dataset.
    season_trend : float, default=0.001
        The long term trend in the seasonality.
    noise : float, default=0.5
        The noise that is applied to the dataset.
    as_frame : bool, default=False
        Whether to return a pandas dataframe instead of a numpy array.
    seed : int | None, default=None
        The seed value for the randomness.
    stack_noise : bool, default=False
        Set the noise to be stacked by a cumulative sum.
    start_date : str | None, default=None
        Also add a start date (only works if `as_frame`=True).

    Returns
    -------
    np.ndarray | pd.DataFrame
        Timeseries dataset with specified characteristics.

    Examples
    --------
    ```py
    from sklego.datasets import make_simpleseries

    make_simpleseries(seed=42)
    # array([-0.34078806, -0.61828731, -0.18458236, ..., -0.27547402,
    #        -0.38237413,  0.13489355])

    make_simpleseries(as_frame=True, start_date="2018-01-01", seed=42).head(3)
    '''
             yt       date
    0 -0.340788 2018-01-01
    1 -0.618287 2018-01-02
    2 -0.184582 2018-01-03
    '''
    ```
    """
    if seed:
        np.random.seed(seed)
    time = np.arange(0, n_samples)
    noise = np.random.normal(0, noise, n_samples)
    if stack_noise:
        noise = noise.cumsum()
    r1, r2 = np.random.normal(0, 1, 2)
    seasonality = r1 * np.sin(time / 365 * 2 * np.pi) + r2 * np.cos(time / 365 * 4 * np.pi + 1)
    result = seasonality + season_trend * seasonality * time + trend * time + noise
    if as_frame:
        if start_date:
            stamps = pd.date_range(start_date, periods=n_samples)
            return pd.DataFrame({"yt": result, "date": stamps})
        return pd.DataFrame({"yt": result})
    return result


def fetch_creditcard(*, cache=True, data_home=None, as_frame=False, return_X_y=False):
    """Load the creditcard dataset. Download it if necessary.

    Note that internally this is using
    [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) from
    scikit-learn, which is experimental.

        ==============   ==============
        Samples total            284807
        Dimensionality               29
        Features                   real
        Target                 int 0, 1
        ==============   ==============

    The datasets contains transactions made by credit cards in September 2013 by european cardholders.
    This dataset present transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
    The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

    Please cite:

        Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
        Calibrating Probability with Undersampling for Unbalanced Classification.
        In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

    Parameters
    ----------
    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.
    data_home : str | None, default=None
        Specify another download and cache folder for the datasets. By default all scikit-learn data is stored in
        '~/scikit_learn_data' subfolders.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with appropriate dtypes (numeric, string or
        categorical). The target is a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a `frame` attribute with the target and the data. If `return_X_y` is True, then
        `(data, target)` will be pandas DataFrames or Series as describe above.
    return_X_y : bool, default=False.
        If True, returns `(data.data, data.target)` instead of a Bunch object.

    Returns
    -------
    Dictionary-like
        With the following attributes:

         * data
            ndarray, shape (284807, 29) if ``as_frame`` is True, ``data`` is a pandas object.
         * target
            ndarray, shape (284807, ) if ``as_frame`` is True, ``target`` is a pandas object.
         * feature_names
            Array of ordered feature names used in the dataset.
         * DESCR
            Description of the creditcard dataset. Best to use print.

    Notes
    -----
    This dataset consists of 284807 samples and 29 features.
    """
    return fetch_openml(
        data_id=1597,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
