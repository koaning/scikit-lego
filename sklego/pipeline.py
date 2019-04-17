'''
Pipelines, variances to the :class:`sklearn.pipeline.Pipeline` object.
'''


import logging
import time

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_memory


def default_log_callback(output, execution_time, **kwargs):
    '''
    The default log callback which logs the step name, shape of the output and
    the execution time of the step.

    :param tuple output:
        (:class:`numpy.ndarray` or :class:`pandas.DataFrame`, \
         :class:`estimator` or :class:`transformer`)
        The output of the step and a step in the pipeline.
    :param float execution_time: The execution time of the step.

    .. note::
        If you write your custom callback function the input is:
            :param function func: The function to be wrapped.
            :param tuple input_args: The input arguments.
            :param dict input_kwargs: The input key-word arguments.
            :param output: The output of the function.
            :param execution_time float: The execution time.
    '''
    logger = logging.getLogger(__name__)
    step_result, step = output
    logger.info(f'[{step}] shape={step_result.shape} '
                f'time={int(execution_time)}s')


def _log_wrapper(log_callback=default_log_callback):
    '''
    Function wrapper to log information after the function is called, about the
    function, input args, input kwargs, output and the execution time.

    :param function log_callback: optional.
        The log callback which is called after `func` is called. Defaults to
        :func:`default_log_callback`. Note, this function should expect the
        same arguments as the default.
    :returns: The function wrapped with a log callback.
    :rtype: function
    '''
    def _(func):
        def _(*args, **kwargs):
            start_time = time.time()
            output = func(*args, **kwargs)
            execution_time = time.time() - start_time
            log_callback(func=func, input_args=args, input_kwargs=kwargs,
                         output=output, execution_time=execution_time)
            return output
        return _
    return _


def _cache_with_function_log_statement(log_callback=default_log_callback):
    '''
    Wraps the `func` with :func:`_log_wrapper` before passing it to
    :method:`_cache`.

    :param function log_callback: optional.
        The log callback function. Defaults to :func:default_log_callback.

    :returns: The cache where its function is wrapped with a log statement.
    '''
    def _(
            self,
            func=None,
            *args,
            **kwargs):
        if callable(func):
            func = _log_wrapper(log_callback)(func)
        return self._cache(func=func, *args, **kwargs)
    return _


class DebugPipeline(Pipeline):
    '''
    A pipeline that has a log statement in between each step, useful for
    debugging.

    :ivar log_callback: optional.
        The callback function that logs information in between each
        intermediate step. Defaults to None. If set to `'default'`,
        :func:`default_log_callback` is used.

        See :func:`default_log_callback` for an example.

    See :class:`sklearn.pipeline.Pipeline` for all other variables.

    .. note::
        This implementation is a hack on the original sklearn Pipeline. It aims
        to have the same behaviour as the original sklearn Pipeline, while
        changing minimal amount of code.

        The log statement is added by overwriting the cache method of the
        memory, such that the function called in the cache is wrapped with a
        functions that calls the log callback function (`log_callback`).

        This hack breaks will break when:
            - The SKlearn pipeline initialization function is changed.
            - The memory is used differently in the fit.
            - The :class:`joblib.memory.Memory` changes behaviour of the
              :func:`cache` method..
            - The :class:`joblib.memory.Memory` starts using a :func:`_cache`
              method.

    :Example:

    >>> # Set-up for example
    >>> import logging
    >>> import sys
    >>>
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> from sklego.pipeline import DebugPipeline
    >>>
    >>> logging.basicConfig(
    ...    format=('[%(funcName)s:%(lineno)d] - %(message)s'),
    ...    level=logging.INFO,
    ...    stream=sys.stdout,
    ... )
    >>>
    >>> # DebugPipeline set-up
    >>> n_samples, n_features = 3, 5
    >>> X = np.zeros((n_samples, n_features))
    >>> y = np.arange(n_samples)
    >>>
    >>> class Adder(TransformerMixin, BaseEstimator):
    ...    def __init__(self, value):
    ...        self._value = value
    ...
    ...    def fit(self, X, y=None):
    ...        return self
    ...
    ...    def transform(self, X):
    ...        return X + self._value
    ...
    ...    def __repr__(self):
    ...        return f'Adder(value={self._value})'
    >>>
    >>> steps = [
    ...     ('add_1', Adder(value=1)),
    ...     ('add_10', Adder(value=10)),
    ...     ('add_100', Adder(value=100)),
    ...     ('add_1000', Adder(value=1000)),
    ... ]
    >>>
    >>> # The DebugPipeline behaves the sames as the Sklearn pipeline.
    >>> pipe = DebugPipeline(steps)
    >>>
    >>> _ = pipe.fit(X, y=y)
    >>> print(pipe.transform(X))
    [[1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]]
    >>>
    >>> # But it has the option to set a `log_callback`, that logs in between
    >>> # each step.
    >>> pipe = DebugPipeline(steps, log_callback='default')
    >>>
    >>> _ = pipe.fit(X, y=y)
    [default_log_callback:34] - [Adder(value=1)] shape=(3, 5) time=0s
    [default_log_callback:34] - [Adder(value=10)] shape=(3, 5) time=0s
    [default_log_callback:34] - [Adder(value=100)] shape=(3, 5) time=0s
    >>> print(pipe.transform(X))
    [[1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]]
    >>>
    >>> # It is possible to set the `log_callback` later then initialisation.
    >>> pipe = DebugPipeline(steps)
    >>> pipe.log_callback = 'default'
    >>>
    >>> _ = pipe.fit(X, y=y)
    [default_log_callback:34] - [Adder(value=1)] shape=(3, 5) time=0s
    [default_log_callback:34] - [Adder(value=10)] shape=(3, 5) time=0s
    [default_log_callback:34] - [Adder(value=100)] shape=(3, 5) time=0s
    >>> print(pipe.transform(X))
    [[1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]]
    >>>
    >>> # It is possible to define your own `log_callback`
    >>> def log_callback(output, execution_time, **kwargs):
    ...     """My custom `log_callback` function
    ...
    ...     Parameters
    ...     ----------
    ...     output : tuple(
    ...             numpy.ndarray or pandas.DataFrame
    ...             :class:estimator or :class:transformer
    ...         )
    ...         The output of the step and a step in the pipeline.
    ...     execution_time : float
    ...         The execution time of the step.  ...
    ...     Note
    ...     ----------
    ...     The **kwargs are for arguments that are not used in this callback.
    ...     """
    ...     logger = logging.getLogger(__name__)
    ...     step_result, step = output
    ...     logger.info(
    ...         f'[{step}] shape={step_result.shape} '
    ...         f'nbytes={step_result.nbytes} time={int(execution_time)}s')
    >>>
    >>> pipe.log_callback = log_callback
    >>>
    >>> _ = pipe.fit(X, y=y)
    [log_callback:20] - [Adder(value=1)] shape=(3, 5) nbytes=120 time=0s
    [log_callback:20] - [Adder(value=10)] shape=(3, 5) nbytes=120 time=0s
    [log_callback:20] - [Adder(value=100)] shape=(3, 5) nbytes=120 time=0s
    >>> print(pipe.transform(X))
    [[1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]]
    >>>
    >>> # Remove the `log_callback` when you want to stop logging.
    >>> pipe.log_callback = None
    >>>
    >>> _ = pipe.fit(X, y=y)
    >>> print(pipe.transform(X))
    [[1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]
     [1111. 1111. 1111. 1111. 1111.]]
    >>>
    >>> # Logging also works with FeatureUnion
    >>> from sklearn.pipeline import FeatureUnion
    >>> pipe_w_default_log_callback = DebugPipeline(steps, log_callback='default')
    >>> pipe_w_custom_log_callback = DebugPipeline(steps, log_callback=log_callback)
    >>>
    >>> pipe_union = DebugPipeline([
    ...     ('feature_union', FeatureUnion([
    ...         ('pipe_w_default_log_callback', pipe_w_default_log_callback),
    ...         ('pipe_w_custom_log_callback', pipe_w_custom_log_callback),
    ...     ])),
    ...     ('final_adder', Adder(10000))
    ... ], log_callback='default')
    >>>
    >>> _ = pipe_union.fit(X, y=y)   # doctest:+ELLIPSIS
    [default_log_callback:34] - [Adder(value=1)] shape=(3, 5) time=0s
    [default_log_callback:34] - [Adder(value=10)] shape=(3, 5) time=0s
    [default_log_callback:34] - [Adder(value=100)] shape=(3, 5) time=0s
    [log_callback:20] - [Adder(value=1)] shape=(3, 5) nbytes=120 time=0s
    [log_callback:20] - [Adder(value=10)] shape=(3, 5) nbytes=120 time=0s
    [log_callback:20] - [Adder(value=100)] shape=(3, 5) nbytes=120 time=0s
    [default_log_callback:34] - [FeatureUnion(...)] shape=(3, 10) time=0s
    >>> print(pipe_union.transform(X))
    [[11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111.]
     [11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111.]
     [11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111.]]
    '''

    def __init__(
            self,
            steps,
            memory=None,
            *,
            log_callback=None):
        self.log_callback = log_callback
        super().__init__(steps=steps, memory=memory)

    @property
    def memory(self):
        # When no log callback function is given, change nothing.
        # Or, if the memory cache was changed, set it back to its original.
        if self._log_callback is None:
            if hasattr(self._memory, '_cache'):
                self._memory.cache = self._memory._cache
            return self._memory

        self._memory = check_memory(self._memory)

        # Overwrite cache function of memory such that it logs the
        # output when the function is called
        if not hasattr(self._memory, '_cache'):
            self._memory._cache = self._memory.cache
        self._memory.cache = _cache_with_function_log_statement(
            self._log_callback).__get__(self._memory, self._memory.__class__)
        return self._memory

    @memory.setter
    def memory(self, memory):
        self._memory = memory

    @property
    def log_callback(self):
        return self._log_callback

    @log_callback.setter
    def log_callback(self, func):
        self._log_callback = func
        if self._log_callback == 'default':
            self._log_callback = default_log_callback
