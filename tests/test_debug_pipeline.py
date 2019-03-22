import pytest

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import (
    OneVsRestClassifier,
    OneVsOneClassifier,
    OutputCodeClassifier,
)
from sklearn.svm import LinearSVC

from sklego.pipeline import DebugPipeline


IRIS = datasets.load_iris()


@pytest.mark.filterwarnings('ignore: The default of the `iid`')  # 0.22
@pytest.mark.filterwarnings('ignore: You should specify a value')  # 0.22
@pytest.mark.parametrize(
    'cls', [OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier])
def test_classifier_gridsearch(cls):
    pipe = DebugPipeline([
        ('ovrc', cls(LinearSVC(random_state=0))),
    ])
    Cs = [0.1, 0.5, 0.8]
    cv = GridSearchCV(pipe, {'ovrc__estimator__C': Cs})
    cv.fit(IRIS.data, IRIS.target)
    best_C = cv.best_estimator_.get_params()['ovrc__estimator__C']
    assert best_C in Cs
