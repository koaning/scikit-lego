import pytest
import pandas as pd
from sklego.preprocessing import ColumnSelector


def test_type_error(random_xy_dataset_regr):
    """For ColumnSelector we require the input data to be a DataFrame. This should fail if this is not the case"""

    # Randomly generate data for a regression
    X, y = random_xy_dataset_regr

    # Try to access column 0 from the input data. This should fail, because it requires a DataFrame as input
    transformer = ColumnSelector([0])
    with pytest.raises(TypeError):
        transformer.fit(X)


def test_key_error(random_xy_dataset_regr):
    """
    For ColumnSelector we require the columns we try to access to be existent in the DataFrame.
    This should fail if this is not the case
    """

    # Randomly generate data for a regression and create a Dataframe
    X, y = random_xy_dataset_regr
    X = pd.DataFrame(X)

    # Try to access a non-existent column in this DataFrame. This should return in a KeyError
    transformer = ColumnSelector(['Non-existent-columnname'])
    with pytest.raises(KeyError):
        transformer.fit(X, y)
