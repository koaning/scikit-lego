import numpy as np
import pandas as pd
import pytest

# from sklearn.datasets import make_classification, make_regression

# from sklearn.dummy import DummyRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.pipeline import Pipeline

# from sklego.common import flatten
# from sklego.meta import HierarchicalClassifier, HierarchicalPredictor, HierarchicalRegressor
# from tests.conftest import general_checks, regressor_checks, classifier_checks, select_tests


@pytest.fixture
def random_xy_grouped_clf_different_classes(request):
    group_size = request.param.get("group_size")
    y_choices_grpa = request.param.get("y_choices_grpa")
    y_choices_grpb = request.param.get("y_choices_grpb")

    np.random.seed(43)
    group_col = np.repeat(["A", "B"], group_size)
    x_col = np.random.normal(size=group_size * 2)
    y_col = np.hstack(
        [
            np.random.choice(y_choices_grpa, size=group_size),
            np.random.choice(y_choices_grpb, size=group_size),
        ]
    )
    df = pd.DataFrame({"group": group_col, "x": x_col, "y": y_col})
    return df
