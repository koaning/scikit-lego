'''
Debug pipeline that has a log statement in between the executed steps.
'''


import logging
import datetime as dt

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_memory


def default_log_callback(output, execution_time):
    '''
    The default log callback which logs the step name, shape of the output and
    the execution time of the step.

    :param tuple output:
        tuple(:class:`numpy.ndarray` or :class:`pandas.DataFrame`,
              :class:`estimator` or :class:`transformer`)
        The output of the step and a step in the pipeline.
    :param float execution_time: The execution time of the step.

    .. note::
        If you write your custom callback function the expected input should
        be the sames as this function.
    '''
    logger = logging.getLogger(__name__)
    step_result, step = output
    logger.info(f'[{step}] shape={step_result.shape} time={execution_time}')


def _log_wrapper(func, log_callback=default_log_callback):
    '''
    Function wrapper to log information after the function is called, about the
    output and the execution time.

    :param function func: The function to be wrapped with a log statement.
    :param function log_callback: optional.
        The log callback which is called after `func` is called. Defaults to
        :func:`default_log_callback`. Note, this function should expect the
        same arguments as the default.

    :returns: The function wrapped with a log callback.
    :rtype: function
    '''
    def _(*args, **kwargs):
        start_time = dt.datetime.now()
        output = func(*args, **kwargs)
        execution_time = dt.datetime.now() - start_time
        log_callback(output, execution_time)
        return output
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
            func = _log_wrapper(func, log_callback=log_callback)
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
