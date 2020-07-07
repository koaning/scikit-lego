import os
import warnings
import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from sklearn.datasets import fetch_openml


def load_penguins(return_X_y=False, as_frame=False, **kwargs):
    """
    Loads the penguins dataset, which is a lovely alternative for the iris dataset. We've
    added this dataset for educational use.

    Data were collected and made available by Dr. Kristen Gorman and the Palmer Station,
    Antarctica LTER, a member of the Long Term Ecological Research Network. The goal
    of the dataset is to predict which species of penguin a penguin belongs to.

    This data originally appeared as a R package and R users can find this data
    in the palmerpenguins package https://github.com/allisonhorst/palmerpenguins.
    You can also go to the repository for some lovely images that explain the dataset.

    To cite this dataset in publications use:

    Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism
    and Environmental Variability within a Community of Antarctic
    Penguins (Genus Pygoscelis). PLoS ONE 9(3): e90081.
    https://doi.org/10.1371/journal.pone.0090081

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param as_frame: give the pandas dataframe instead of X, y matrices (default=False)
    :param give_pandas: Deprecated since version 0.5.0. Please use as_frame instead.

    :Example:
    >>> from sklego.datasets import load_penguins
    >>> X, y = load_penguins(return_X_y=True)
    >>> X.shape
    (344, 6)
    >>> y.shape
    (344,)
    >>> load_penguins(as_frame=True).columns
    Index(['species', 'island', 'bill_length_mm', 'bill_depth_mm',
       'flipper_length_mm', 'body_mass_g', 'sex'],
      dtype='object')

    Additional data use information
    ###############################

    Anyone interested in publishing the data should contact
    `Dr. Kristen Gorman <https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php>`_
    about analysis and working together on any final products.

    From Gorman et al. (2014):

    > “Data reported here are publicly available within the PAL-LTER data
    > system (datasets \#219, 220, and 221):
    > `<http://oceaninformatics.ucsd.edu/datazoo/data/pallter/datasets>`_.
    > Individuals interested in using these data are therefore expected to
    > follow the US LTER Network’s Data Access Policy, Requirements and Use
    > Agreement: `<https://lternet.edu/data-access-policy/>`_.”

    **Please cite data using the following:**

    **Adélie penguins:**

      - Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size
        measurements and isotopic signatures of foraging among adult male
        and female Adélie penguins (*Pygoscelis adeliae*) nesting along the
        Palmer Archipelago near Palmer Station, 2007-2009 ver 5.
        Environmental Data Initiative.
        `<https://doi.org/10.6073/pasta/98b16d7d563f265cb52372c8ca99e60f>`_
        (Accessed 2020-06-08).

    **Gentoo penguins:**

      - Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size
        measurements and isotopic signatures of foraging among adult male
        and female Gentoo penguin (*Pygoscelis papua*) nesting along the
        Palmer Archipelago near Palmer Station, 2007-2009 ver 5.
        Environmental Data Initiative.
        `<https://doi.org/10.6073/pasta/7fca67fb28d56ee2ffa3d9370ebda689>`_
        (Accessed 2020-06-08).

    **Chinstrap penguins:**

      - Palmer Station Antarctica LTER and K. Gorman, 2020. Structural size
        measurements and isotopic signatures of foraging among adult male
        and female Chinstrap penguin (*Pygoscelis antarcticus*) nesting
        along the Palmer Archipelago near Palmer Station, 2007-2009 ver 6.
        Environmental Data Initiative.
        `<https://doi.org/10.6073/pasta/c14dfcfada8ea13a17536e73eb6fbe9e>`_
        (Accessed 2020-06-08).
    """
    if "give_pandas" in kwargs:
        warnings.warn(
            "give_pandas is deprecated since version 0.5.0 and will be removed in version 0.7.0. "
            "Please use as_frame instead.",
            FutureWarning,
        )
        as_frame = kwargs["give_pandas"]

    filepath = resource_filename("sklego", os.path.join("data", "penguins.zip"))
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]].values,
        df["species"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_arrests(return_X_y=False, as_frame=False, **kwargs):
    """
    Loads the arrests dataset which can serve as a benchmark for fairness. It is data on
    the police treatment of individuals arrested in Toronto for simple possession of small
    quantities of marijuana. The goal is to predict whether or not the arrestee was released
    with a summons while maintaining a degree of fairness.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param as_frame: give the pandas dataframe instead of X, y matrices (default=False)
    :param give_pandas: Deprecated since version 0.5.0. Please use as_frame instead.

    :Example:
    >>> from sklego.datasets import load_arrests
    >>> X, y = load_arrests(return_X_y=True)
    >>> X.shape
    (5226, 7)
    >>> y.shape
    (5226,)
    >>> load_arrests(as_frame=True).columns
    Index(['released', 'colour', 'year', 'age', 'sex', 'employed', 'citizen',
           'checks'],
          dtype='object')

    The dataset was copied from the carData R package and can originally be found in:

    - Personal communication from Michael Friendly, York University.

    The documentation page of the dataset from the package can be viewed here:
    http://vincentarelbundock.github.io/Rdatasets/doc/carData/Arrests.html
    """
    if "give_pandas" in kwargs:
        warnings.warn(
            "give_pandas is deprecated since version 0.5.0 and will be removed in version 0.7.0. "
            "Please use as_frame instead.",
            FutureWarning,
        )
        as_frame = kwargs["give_pandas"]

    filepath = resource_filename("sklego", os.path.join("data", "arrests.zip"))
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[["colour", "year", "age", "sex", "employed", "citizen", "checks"]].values,
        df["released"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_chicken(return_X_y=False, as_frame=False, **kwargs):
    """
    Loads the chicken dataset. The chicken data has 578 rows and 4 columns
    from an experiment on the effect of diet on early growth of chicks.
    The body weights of the chicks were measured at birth and every second
    day thereafter until day 20. They were also measured on day 21.
    There were four groups on chicks on different protein diets.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param as_frame: give the pandas dataframe instead of X, y matrices (default=False)
    :param give_pandas: Deprecated since version 0.5.0. Please use as_frame instead.

    :Example:

    >>> from sklego.datasets import load_chicken
    >>> X, y = load_chicken(return_X_y=True)
    >>> X.shape
    (578, 3)
    >>> y.shape
    (578,)
    >>> load_chicken(as_frame=True).columns
    Index(['weight', 'time', 'chick', 'diet'], dtype='object')

    The datasets can be found in the following sources:

    - Crowder, M. and Hand, D. (1990), Analysis of Repeated Measures, Chapman and Hall (example 5.3)
    - Hand, D. and Crowder, M. (1996), Practical Longitudinal Data Analysis, Chapman and Hall (table A.2)
    """
    if "give_pandas" in kwargs:
        warnings.warn(
            "give_pandas is deprecated since version 0.5.0 and will be removed in version 0.7.0. "
            "Please use as_frame instead.",
            FutureWarning,
        )
        as_frame = kwargs["give_pandas"]

    filepath = resource_filename("sklego", os.path.join("data", "chickweight.zip"))
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = df[["time", "diet", "chick"]].values, df["weight"].values
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_abalone(return_X_y=False, as_frame=False, **kwargs):
    """
    Loads the abalone dataset where the goal is to predict the gender of the creature.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param as_frame: give the pandas dataframe instead of X, y matrices (default=False)
    :param give_pandas: Deprecated since version 0.5.0. Please use as_frame instead.

    :Example:

    >>> from sklego.datasets import load_abalone
    >>> X, y = load_abalone(return_X_y=True)
    >>> X.shape
    (4177, 8)
    >>> y.shape
    (4177,)
    >>> load_abalone(as_frame=True).columns
    Index(['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
           'viscera_weight', 'shell_weight', 'rings'],
          dtype='object')

    The dataset was copied from Kaggle and can originally be found in: can be found in the following sources:

    - Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994)
    "The Population Biology of Abalone (_Haliotis_ species) in Tasmania."
    Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288)
    """
    if "give_pandas" in kwargs:
        warnings.warn(
            "give_pandas is deprecated since version 0.5.0 and will be removed in version 0.7.0. "
            "Please use as_frame instead.",
            FutureWarning,
        )
        as_frame = kwargs["give_pandas"]

    filepath = resource_filename("sklego", os.path.join("data", "abalone.zip"))
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
    ].values
    y = df["sex"].values
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_heroes(return_X_y=False, as_frame=False, **kwargs):
    """
    A dataset from a video game: "heroes of the storm". The goal of the dataset
    is to predict the attack type. Note that the pandas dataset returns more information.
    This is because we wanted to keep the X simple in the return_X_y case.
    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param as_frame: give the pandas dataframe instead of X, y matrices (default=False)
    :param give_pandas: Deprecated since version 0.5.0. Please use as_frame instead.

    :Example:
    >>> X, y = load_heroes(return_X_y=True)
    >>> X.shape
    (84, 2)
    >>> y.shape
    (84,)
    >>> df = load_heroes(as_frame=True)
    >>> df.columns
    Index(['name', 'attack_type', 'role', 'health', 'attack', 'attack_spd'], dtype='object')
    """
    if "give_pandas" in kwargs:
        warnings.warn(
            "give_pandas is deprecated since version 0.5.0 and will be removed in version 0.7.0. "
            "Please use as_frame instead.",
            FutureWarning,
        )
        as_frame = kwargs["give_pandas"]

    filepath = resource_filename("sklego", os.path.join("data", "heroes.zip"))
    df = pd.read_csv(filepath)
    if as_frame:
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
    as_frame=False,
    seed=None,
    stack_noise=False,
    start_date=None,
    **kwargs
):
    """
    Generate a very simple timeseries dataset to play with. The generator
    assumes to generate daily data with a season, trend and noise.

    :param n_samples: The number of days to simulate the timeseries for.
    :param trend: The long term trend in the dataset.
    :param season_trend: The long term trend in the seasonality.
    :param noise: The noise that is applied to the dataset.
    :param as_frame: Return a pandas dataframe instead of a numpy array.
    :param seed: The seed value for the randomness.
    :param stack_noise: Set the noise to be stacked by a cumulative sum.
    :param start_date: Also add a start date (only works if `as_frame`=True).
    :param give_pandas: Deprecated since version 0.5.0. Please use as_frame instead.
    :return: numpy array unless dataframe is specified

    :Example:

    >>> from sklego.datasets import make_simpleseries
    >>> make_simpleseries(seed=42)
    array([-0.34078806, -0.61828731, -0.18458236, ..., -0.27547402,
           -0.38237413,  0.13489355])
    >>> make_simpleseries(as_frame=True, start_date="2018-01-01", seed=42).head(3)
             yt       date
    0 -0.340788 2018-01-01
    1 -0.618287 2018-01-02
    2 -0.184582 2018-01-03
    """
    if "give_pandas" in kwargs:
        warnings.warn(
            "give_pandas is deprecated since version 0.5.0 and will be removed in version 0.7.0. "
            "Please use as_frame instead.",
            FutureWarning,
        )
        as_frame = kwargs["give_pandas"]

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
    if as_frame:
        if start_date:
            stamps = pd.date_range(start_date, periods=n_samples)
            return pd.DataFrame({"yt": result, "date": stamps})
        return pd.DataFrame({"yt": result})
    return result


def load_hearts(return_X_y=False, as_frame=False, **kwargs):
    """
    Loads the Cleveland Heart Diseases dataset. The goal is to predict the presence of a heart disease (target values 1, 2, 3, and 4).
    The data originates from research to heart diseases by four institutions and originally contains 76 attributes.
    Yet, all published experiments refer to using a subset of 13 features and one target.
    This implementation loads the Cleveland dataset of the research which is the only set used by ML researchers to this date.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param as_frame: give the pandas dataframe instead of X, y matrices (default=False)
    :param give_pandas: Deprecated since version 0.5.0. Please use as_frame instead.

    :Example:
    >>> X, y = load_hearts(return_X_y=True)
    >>> X.shape
    (303, 13)
    >>> y.shape
    (303,)
    >>> df = load_hearts(as_frame=True)
    >>> df.columns
    Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
          dtype='object')

    The dataset can originally be found here:
    https://archive.ics.uci.edu/ml/datasets/Heart+Disease

    The responsible institutions for the entire database are:

    1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
    2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
    3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
    4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

    The documentation of the dataset can be viewed here:
    https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
    """
    if "give_pandas" in kwargs:
        warnings.warn(
            "give_pandas is deprecated since version 0.5.0 and will be removed in version 0.7.0. "
            "Please use as_frame instead.",
            FutureWarning,
        )
        as_frame = kwargs["give_pandas"]

    filepath = resource_filename("sklego", os.path.join("data", "hearts.zip"))
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
    ].values
    y = df["target"].values
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def fetch_creditcard(*, cache=True, data_home=None, as_frame=False, return_X_y=False):
    """
    Load the creditcard dataset. Download it if necessary.

    Note that internally this is using `fetch_openml` from scikit-learn, which is experimental.

    ==============   ==============
    Samples total            284807
    Dimensionality               29
    Features                   real
    Target                 int 0, 1
    ==============   ==============

    The datasets contains transactions made by credit cards in September 2013 by european
    cardholders. This dataset present transactions that occurred in two days, where we have
    492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive
    class (frauds) account for 0.172% of all transactions.

    Please cite:
        Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
        Calibrating Probability with Undersampling for Unbalanced Classification.
        In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

    :param version: integer or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.
    :param cache: boolean, default=True
        Whether to cache downloaded datasets using joblib.
    :param data_home: optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
    :param as_frame: boolean, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.
    :param return_X_y: : boolean, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

    Returns
    -------
    :return:
        Dictionary-like object, with the following attributes.

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
