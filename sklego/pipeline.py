'''
Debug pipeline that has a log statement in between the executed steps.
'''


import logging
import datetime as dt

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_memory


def _default_log_callback(output, execution_time):
    '''The default log callback which logs the step name, shape of the output
    and the execution time of the step.

    Parameters
    ----------
    output : tuple(
            numpy.ndarray or pandas.DataFrame
            :class:estimator or :class:transformer
        )
        The output of the step and a step in the pipeline.
    execution_time : float
        The execution time of the step.

    Note
    ----------
    If you write your custom callback function the expected input should be the
    sames as this function.
    '''
    logger = logging.getLogger(__name__)
    step_result, step = output
    logger.info(f'[{step}] shape={step_result.shape} time={execution_time}')


def _log_wrapper(func, log_callback=_default_log_callback):
    '''Function wrapper to log information after the function is called, about
    the output and the execution time.

    Parameters
    ----------
    func : function
        The function to be wrapped with a log statement.
    log_callback : function, optional
        The log callback which is called after `func` is called. Defaults to
        :func:_default_log_callback. Note, this function should expect the
        same arguments as the default.

    Returns
    ----------
    function : The function wrapped with a log callback.
    '''
    def _(*args, **kwargs):
        start_time = dt.datetime.now()
        output = func(*args, **kwargs)
        execution_time = dt.datetime.now() - start_time
        log_callback(output, execution_time)
        return output
    return _


def _cache_with_function_log_statement(log_callback=_default_log_callback):
    '''Wraps the `func` with :func:_log_wrapper before passing it to
    :method:_cache.

    Parameters
    ----------
    log_callback : function, optional
        The log callback function. Defaults to :func:_default_log_callback.

    Returns
    ----------
    function : The cache where its function is wrapped with a log statement.
    '''
    def _(
            self,
            func=None,
            *args,
            **kwargs):
        if callable(func):
            func = _log_wrapper(func, log_callback=log_callback)
        return self._cache(func=func, *args, **kwargs)
    return _


class Pipeline(Pipeline):
    '''A pipeline that has a log statement in between each step, useful for
    debugging.

    Parameters
    ----------
    log_callback : function, optional
        The callback function that logs information in between each
        intermediate step. Defaults to None. If set to `'default'`,
        :func:_default_log_callback is used.

        See :func:_default_log_callback for an example.

    See :class:sklearn.pipeline.PipeLine for all other parameters.

    Note
    ----------
    This implementation is a hack on the original sklearn Pipeline. It aims to
    have the same behaviour as the original sklearn Pipeline, while changing
    minimal amount of code.

    The log statement is added by overwriting the cache method of the memory,
    such that the function called in the cache is wrapped with a functions that
    calls the log callback function (`log_callback`).

    This hack breaks will break when:
    1) The SKlearn pipeline initialization function is changed.
    2) The memory is used differently in the fit.
    3) The :class:joblib.memory.Memory changes behaviour of the cache function.
    4) The :class:joblib.memory.Memory starts using a `_cache` method.
    '''

    def __init__(
            self,
            steps,
            memory=None,
            *,
            log_callback=None):
        self.memory = check_memory(memory)
        self.log_callback = log_callback
        super().__init__(steps=steps, memory=self.memory)

    @property
    def log_callback(self):
        return self._log_callback

    @log_callback.setter
    def log_callback(self, func):
        self._log_callback = func
        if self._log_callback == 'default':
            self._log_callback = _default_log_callback

        # When no log callback function is given, change nothing.
        # Or, if the memory cache was changed, set it back to its original.
        if self._log_callback is None:
            if hasattr(self.memory, '_cache'):
                self.memory.cache = self.memory._cache
            return

        # Overwrite cache function of memory such that it logs the
        # output when the function is called
        if not hasattr(self.memory, '_cache'):
            self.memory._cache = self.memory.cache
        self.memory.cache = _cache_with_function_log_statement(
            self._log_callback).__get__(self.memory, self.memory.__class__)
