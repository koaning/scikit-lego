import os
import pandas as pd


def load_chicken():
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, "data", "chickweight.csv")
    return pd.read_csv(filepath)
