from sklego.preprocessing import ColumnCapper
from sklearn.base import TransformerMixin
import pandas as pd


test_data = pd.DataFrame({'var1':range(101)})
capper = ColumnCapper(quantile_thresholds=0.9)

def test_capper():
    assert (capper.transform(test_data).iloc[91]['var1'] == 90.0)


if __name__ == "__main__":
    test_capper()
    print("Everything passed")