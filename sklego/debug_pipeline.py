'''
Debug pipeline that logs the in-between steps of the pipeline.
'''


import logging
import inspect
import datetime as dt
from functools import wraps

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_memory


def _default_log_callback(output, execution_time):
    '''The default log callback, that logs the step name, shape of the output
    and the execution time of the step.

    Parameters
    ----------
    output : Tuple(
            numpy.ndarray or pandas.DataFrame
            sklearn.base.BaseEstimator or sklearn.base.TransformerMixin
        )
        The output of the step and a step in the pipeline.
    execution_time : float
        The execution time of the step.
    '''
    logger = logging.getLogger(__name__)
    step_result, step = output
    logger.info(f'[{step}] shape={step_result.shape} time={execution_time}')


def _log_wrapper(func, log_callback=_default_log_callback):
    '''Function wrapper to log information after the function is called, about
    the output ant the execution time.

    Parameters
    ----------
    func : function
        The function to be wrapped with a log statement.
    log_callback : function, optional
        The log callback. Defaults to :func:_default_log_callback. It should
        expect the same arguments as the default.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(inspect.stack()[1].function)

        tic = dt.datetime.now()
        out, step = func(*args, **kwargs)
        toc = str(dt.datetime.now() - tic)

        logger.info(f'[{step}] shape={out.shape} time={toc}')
        return out, step
    return wrapper


def _log_step_cache(self, func=None, *args, **kwargs):
    '''Wraps the `func` with a log step, then uses the original case.

    Parameters
    ----------
    func : function
        Function to be wrapper with the :func:_default_log_step. Defaults to
        `None`.

    Returns
    ---------
    See :method:self._cache.
    '''
    if callable(func):
        func = _default_log_step(func)
    return self._cache(func=func, *args, **kwargs)


class DebugPipeline(Pipeline):
    '''A pipeline that has a log statement in-between each step, useful for
    debugging.

    This implementation is a hack on the original sklearn Pipeline. It aims to
    maintain the same behaviour as the original sklearn Pipeline, while
    changing minimal amount of code.

    The log statement is added by overwriting the cache method of the memory,
    such that the function called in the cache is wrapped with a functions that
    has the log statement. For an example wrapper function see
    :func:_default_log_step.

    See :class:sklearn.pipeline.PipeLine for all other information.
    '''

    def __init__(self, steps, memory=None):
        # Overwrite cache function of memory such that it logs the output when
        # the function is called
        memory = check_memory(memory)
        memory._cache = memory.cache
        memory.cache = _log_step_cache.__get__(memory, memory.__class__)

        super().__init__(steps, memory=memory)
