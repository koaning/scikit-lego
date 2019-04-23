import os
import pandas as pd


def load_chicken(give_pandas=False):
    """
    Loads the chicken dataset. The chicken data has 578 rows and 4 columns
    from an experiment on the effect of diet on early growth of chicks.
    The body weights of the chicks were measured at birth and every second
    day thereafter until day 20. They were also measured on day 21.
    There were four groups on chicks on different protein diets.

    :param give_pandas: give the pandas dataframe instead of X, y matrices (default=False)
    :return: (**X**, **y**) unless dataframe is returned

    :Example:

    >>> from sklego.datasets import load_chicken
    >>> X, y = load_chicken()
    >>> X.shape
    (578, 3)
    >>> y.shape
    (578,)
    >>> load_chicken(give_pandas=True).columns
    Index(['weight', 'time', 'chick', 'diet'], dtype='object')

    The datasets can be found in the folowing sources:

    - Crowder, M. and Hand, D. (1990), Analysis of Repeated Measures, Chapman and Hall (example 5.3)
    - Hand, D. and Crowder, M. (1996), Practical Longitudinal Data Analysis, Chapman and Hall (table A.2)
    """
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, "data", "chickweight.csv")
    df = pd.read_csv(filepath)
    if give_pandas:
        return df
    return df[['time', 'diet', 'chick']].values, df['weight'].values
