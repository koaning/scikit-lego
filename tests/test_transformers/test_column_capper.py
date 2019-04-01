from sklego.preprocessing import ColumnCapper
import pandas as pd


def test_capper_df():
    test_data = pd.DataFrame({'var1': [1, 2, 3, 4, 5],
                              'var2': [2, 3, 4, 5, 6]})
    capper = ColumnCapper(thresholds=0.8)
    # write proper test here this one is a standin
    assert capper.fit(test_data).transform(test_data).shape == test_data.shape
