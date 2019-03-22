import logging
import pytest

from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import (
    OneVsRestClassifier,
    OneVsOneClassifier,
    OutputCodeClassifier,
)
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from sklego.pipeline import DebugPipeline


IRIS = datasets.load_iris()


class Adder(TransformerMixin, BaseEstimator):
    def __init__(self, value):
        self._value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X + self._value

    def __repr__(self):
        return f'Adder(value={self._value})'


def custom_log_callback(output, execution_time, **kwargs):
    '''My custom `log_callback` function

    Parameters
    ----------
    output : tuple(
            numpy.ndarray or pandas.DataFrame
            :class:estimator or :class:transformer
        )
        The output of the step and a step in the pipeline.
    execution_time : float
        The execution time of the step.
    '''
    logger = logging.getLogger(__name__)
    step_result, step = output
    logger.info(f'[{step}] shape={step_result.shape} '
                f'nbytes={step_result.nbytes} time={execution_time}')


@pytest.fixture
def steps():
    return [
        ('add_1', Adder(value=1)),
        ('add_10', Adder(value=10)),
        ('add_100', Adder(value=100)),
        ('add_1000', Adder(value=1000)),
    ]


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


def test_no_logs_when_log_callback_is_None(caplog, steps):
    pipe = DebugPipeline(steps, log_callback=None)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        pipe.fit(IRIS.data, IRIS.target)
    assert not caplog.text, f'Log should be empty: {caplog.text}'


def test_output_shape_in_logs_when_log_callback_is_default(caplog, steps):
    pipe = DebugPipeline(steps, log_callback='default')
    caplog.clear()
    with caplog.at_level(logging.INFO):
        pipe.fit(IRIS.data, IRIS.target)
    assert caplog.text, f'Log should be none empty: {caplog.text}'
    shape_str = f'shape={IRIS.data.shape}'
    assert shape_str in caplog.text, f'"{shape_str}" should be in {caplog.text}'
    assert caplog.text.count(shape_str) == (len(pipe.steps) - 1), \
        f'"{shape_str}" should be {len(pipe.steps) - 1} times in {caplog.text}'


def test_time_in_logs_when_log_callback_is_default(caplog, steps):
    pipe = DebugPipeline(steps, log_callback='default')
    caplog.clear()
    with caplog.at_level(logging.INFO):
        pipe.fit(IRIS.data, IRIS.target)
    assert caplog.text, f'Log should be none empty: {caplog.text}'
    assert f'time=' in caplog.text, f'"time=" should be in: {caplog.text}'
    assert caplog.text.count('time') == (len(pipe.steps) - 1), \
        f'"time" should be {len(pipe.steps) - 1} times in {caplog.text}'


def test_step_name_in_logs_when_log_callback_is_default(caplog, steps):
    pipe = DebugPipeline(steps, log_callback='default')
    caplog.clear()
    with caplog.at_level(logging.INFO):
        pipe.fit(IRIS.data, IRIS.target)
    assert caplog.text, f'Log should be none empty: {caplog.text}'
    for _, step in pipe.steps[:-1]:
        assert str(step) in caplog.text, f'{step} should be in: {caplog.text}'
        assert caplog.text.count(str(step)) == 1, \
            f'{step} should be once in {caplog.text}'


def test_nbytes_in_logs_when_log_callback_is_custom(caplog, steps):
    pipe = DebugPipeline(steps, log_callback=custom_log_callback)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        pipe.fit(IRIS.data, IRIS.target)
    assert caplog.text, f'Log should be none empty: {caplog.text}'
    assert 'nbytes=' in caplog.text, f'"nbytes=" should be in: {caplog.text}'
    assert caplog.text.count('nbytes=') == (len(pipe.steps) - 1), \
        f'"nbytes=" should be {len(pipe.steps) - 1} times in {caplog.text}'


def test_feature_union(caplog, steps):
    pipe_w_default_log_callback = DebugPipeline(steps, log_callback='default')
    pipe_w_custom_log_callback = DebugPipeline(
        steps, log_callback=custom_log_callback)

    pipe_union = FeatureUnion([
        ('pipe_w_default_log_callback', pipe_w_default_log_callback),
        ('pipe_w_custom_log_callback', pipe_w_custom_log_callback),
    ])

    caplog.clear()
    with caplog.at_level(logging.INFO):
        pipe_union.fit(IRIS.data, IRIS.target)
    assert caplog.text, f'Log should be none empty: {caplog.text}'
    for pipe in [pipe_w_default_log_callback, pipe_w_custom_log_callback]:
        for _, step in pipe.steps[:-1]:
            assert str(step) in caplog.text, \
                f'{step} should be in: {caplog.text}'
            assert caplog.text.count(str(step)) == 2, \
                f'{step} should be once in {caplog.text}'
