import pytest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklego.common import flatten
from sklego.meta import Thresholder
from tests.conftest import general_checks, classifier_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, classifier_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_fit2d_predict1d",
            "check_methods_subset_invariance",
            "check_dont_overwrite_parameters",
            "check_classifiers_classes",
            "check_classifiers_train",
            "check_supervised_y_2d",
            "check_classifier_data_not_an_array", # https://github.com/koaning/scikit-lego/issues/490
        ]
        # outliers train wont work because we have two thresholds
    ),
)
def test_standard_checks(test_fn):
    trf = Thresholder(LogisticRegression(), threshold=0.5)
    test_fn(Thresholder.__name__, trf)


def test_same_threshold():
    mod1 = Thresholder(LogisticRegression(), threshold=0.5)
    mod2 = LogisticRegression()
    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0
    assert (mod1.fit(X, y).predict(X) == mod2.fit(X, y).predict(X)).all()


def test_diff_threshold():
    mod1 = Thresholder(LogisticRegression(), threshold=0.5)
    mod2 = Thresholder(LogisticRegression(), threshold=0.7)
    mod3 = Thresholder(LogisticRegression(), threshold=0.9)
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0
    assert mod1.fit(X, y).predict(X).sum() >= mod2.fit(X, y).predict(X).sum()
    assert mod2.fit(X, y).predict(X).sum() >= mod3.fit(X, y).predict(X).sum()


def test_raise_error1():
    with pytest.raises(ValueError):
        # we only support classification models
        mod = Thresholder(LinearRegression(), threshold=0.7)
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 3))
        y = np.random.normal(0, 1, (100,)) < 0
        mod.fit(X, y)


def test_raise_error2():
    with pytest.raises(ValueError):
        mod = Thresholder(LinearRegression(), threshold=0.7)
        np.random.seed(42)
        X = np.random.normal(0, 1, (1000, 3))
        # we only support two classes
        y = np.random.choice(["a", "b", "c"], 1000)
        mod.fit(X, y)


def test_refit_always():
    lr = LogisticRegression()
    mod = Thresholder(lr, threshold=0.5, refit=True)
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0
    assert mod.fit(X, y).predict(X).shape == y.shape


def test_refit_auto():
    lr = LogisticRegression()
    mod = Thresholder(lr, threshold=0.5, refit=False)
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0
    mod.fit(X, y).predict(X)
    assert mod.fit(X, y).predict(X).shape == y.shape


@pytest.mark.parametrize('refit', [True, False])
def test_passes_sample_weight(refit):
    class CustomLR(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            assert sample_weight is not None
            super().fit(X, y)

    mod = Thresholder(CustomLR(), threshold=0.5, refit=refit)
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0
    weight = np.random.random(100)

    mod.fit(X, y, sample_weight=weight)


def test_no_refit_does_not_fit_underlying():
    X = np.array([1, 2, 3, 4]).reshape(-1, 1)
    y_ones = np.array([0, 1, 1, 1]).reshape(-1, )
    y_zeros = np.array([0, 0, 0, 1]).reshape(-1, )

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X, y_ones)
    a = Thresholder(clf, threshold=0.2, refit=False)      
    a.fit(X, y_zeros)

    assert a.predict(np.array([[1]])) == 1


def test_refit_fits_underlying():
    X = np.array([1, 2, 3, 4]).reshape(-1, 1)
    y_ones = np.array([0, 1, 1, 1]).reshape(-1, )
    y_zeros = np.array([0, 0, 0, 1]).reshape(-1, )

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X, y_ones)
    a = Thresholder(clf, threshold=0.2, refit=True)      
    a.fit(X, y_zeros)

    assert a.predict(np.array([[1]])) == 0


def test_stacking_classifier():
    '''
    Tests issue https://github.com/koaning/scikit-lego/issues/501

    No asserts are added as we only test for being exception free.
    When cloning the model in Thresholder an unfitted model is generated
    where no predict_proba exists
    '''
    estimators = [("dummy", DummyClassifier(strategy="constant", constant=0))]

    X = np.random.normal(0, 1, (100, 3))
    y = np.random.normal(0, 1, (100,)) < 0

    clf = StackingClassifier(estimators=estimators, final_estimator=DummyClassifier(strategy="constant", constant=0))

    clf.fit(X, y)

    a = Thresholder(clf, threshold=0.2)
    a.fit(X, y)
    a.predict(X)


def test_nans_could_work():
    X = np.array([[np.nan, 4], [7, 3], [5, 5], [7, 2], [5, 7]])
    y = np.array([1, 0, 1, 0, 1])
    model = Thresholder(HistGradientBoostingClassifier(), 0.6)
    model.fit(X, y)
