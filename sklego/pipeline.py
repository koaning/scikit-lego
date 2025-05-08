"""
Pipelines, variances to the `sklearn.pipeline.Pipeline` object.
"""

import logging
import time

from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.utils.validation import check_memory


def default_log_callback(output, execution_time, **kwargs):
    """The default log callback which logs the step name, shape of the output and the execution time of the step.

    !!! info

        If you write your custom callback function the input is:

        | Parameter        | Type             | Description                    |
        | ---------------- | ---------------- | ------------------------------ |
        | `func`           | Callable[..., T] | The function to be wrapped     |
        | `input_args`     | tuple            | The input arguments            |
        | `input_kwargs`   | dict             | The input key-word arguments   |
        | `output`         | T                | The output of the function     |
        | `execution_time` | float            | The execution time of the step |

    Parameters
    ----------
    output : tuple[np.ndarray | pd.DataFrame, estimator | transformer]
        The output of the step and a step in the pipeline.
    execution_time : float
        The execution time of the step.
    """
    logger = logging.getLogger(__name__)
    step_result, step = output
    logger.info(f"[{step}] shape={step_result.shape} time={int(execution_time)}s")


def _log_wrapper(log_callback=default_log_callback):
    """Function wrapper to log information after the function is called, about the function, input args, input kwargs,
    output and the execution time.

    Parameters
    ----------
    log_callback : Callable, default=default_log_callback
        The log callback which is called after `func` is called. Note, this function should expect the same arguments
        as the default.

    Returns
    -------
    Callable
        The function wrapped with a log callback.
    """

    def _(func):
        def _(*args, **kwargs):
            start_time = time.time()
            output = func(*args, **kwargs)
            execution_time = time.time() - start_time
            log_callback(
                func=func,
                input_args=args,
                input_kwargs=kwargs,
                output=output,
                execution_time=execution_time,
            )
            return output

        return _

    return _


def _cache_with_function_log_statement(log_callback=default_log_callback):
    """Wraps the `func` with `_log_wrapper` before passing it to `_cache`.

    Parameters
    ----------
    log_callback : Callable, default=default_log_callback
        The log callback function.

    Returns
    -------
    Callable
        The function wrapped with a log callback.
    """

    def _(self, func=None, *args, **kwargs):
        if callable(func):
            func = _log_wrapper(log_callback)(func)
        return self._cache(func=func, *args, **kwargs)

    return _


class DebugPipeline(Pipeline):
    """A pipeline that has a log statement in between each step, useful for debugging purposes.

    See [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline)
    for all other parameters other than `log_callback`.

    !!! note

        This implementation is a hack on the original sklearn Pipeline. It aims to have the same behaviour as the
        original sklearn Pipeline, while changing minimal amount of code.

        The log statement is added by overwriting the cache method of the memory, such that the function called in the
        cache is wrapped with a functions that calls the log callback function (`log_callback`).

        This hack will break when:

        - The sklearn pipeline initialization function is changed.
        - The memory is used differently in the fit.
        - The [`joblib.memory.Memory`](https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html)
            changes behaviour of the `cache` method.
        - The [`joblib.memory.Memory`](https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html)
            starts using a `_cache` method.

    Parameters
    ----------
    log_callback : Callable | None, default=None
        The callback function that logs information in between each intermediate step.
        If set to `"default"`, `default_log_callback` is used.

    Examples
    --------
    ```py
    # Set-up for example
    import logging
    import sys

    import numpy as np
    import pandas as pd

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklego.pipeline import DebugPipeline

    logging.basicConfig(
        format=("[%(funcName)s:%(lineno)d] - %(message)s"),
        level=logging.INFO,
        stream=sys.stdout,
        )

    # DebugPipeline set-up
    n_samples, n_features = 3, 5
    X = np.zeros((n_samples, n_features))
    y = np.arange(n_samples)

    class Adder(TransformerMixin, BaseEstimator):
        def __init__(self, value):
            self._value = value

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X + self._value

        def __repr__(self):
            return f"Adder(value={self._value})"

    steps = [
        ("add_1", Adder(value=1)),
        ("add_10", Adder(value=10)),
        ("add_100", Adder(value=100)),
        ("add_1000", Adder(value=1000)),
    ]

    # The DebugPipeline behaves the sames as the Sklearn pipeline.
    pipe = DebugPipeline(steps)

    _ = pipe.fit(X, y=y)
    print(pipe.transform(X))
    # [[1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]]

    # But it has the option to set a `log_callback`, that logs in between each step.
    pipe = DebugPipeline(steps, log_callback="default")

    _ = pipe.fit(X, y=y)
    # [default_log_callback:34] - [Adder(value=1)] shape=(3, 5) time=0s
    # [default_log_callback:34] - [Adder(value=10)] shape=(3, 5) time=0s
    # [default_log_callback:34] - [Adder(value=100)] shape=(3, 5) time=0s

    print(pipe.transform(X))
    # [[1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]]

    # It is possible to set the `log_callback` later then initialisation.
    pipe = DebugPipeline(steps)
    pipe.log_callback = "default"

    _ = pipe.fit(X, y=y)
    # [default_log_callback:34] - [Adder(value=1)] shape=(3, 5) time=0s
    # [default_log_callback:34] - [Adder(value=10)] shape=(3, 5) time=0s
    # [default_log_callback:34] - [Adder(value=100)] shape=(3, 5) time=0s

    print(pipe.transform(X))
    # [[1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]]

    # It is possible to define your own `log_callback` function.
    def log_callback(output, execution_time, **kwargs):
        '''My custom `log_callback` function

        Parameters
        output : tuple[np.ndarray | pd.DataFrame, estimator | transformer]
            The output of the step and a step in the pipeline.
        execution_time : float
            The execution time of the step.

        Note
        The **kwargs are for arguments that are not used in this callback.
        '''
        logger = logging.getLogger(__name__)
        step_result, step = output
        logger.info(
            f"[{step}] shape={step_result.shape} "
            f"nbytes={step_result.nbytes} time={int(execution_time)}s")

    pipe.log_callback = log_callback

    _ = pipe.fit(X, y=y)
    # [log_callback:20] - [Adder(value=1)] shape=(3, 5) nbytes=120 time=0s
    # [log_callback:20] - [Adder(value=10)] shape=(3, 5) nbytes=120 time=0s
    # [log_callback:20] - [Adder(value=100)] shape=(3, 5) nbytes=120 time=0s

    print(pipe.transform(X))
    # [[1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]]

    # Remove the `log_callback` when you want to stop logging.
    pipe.log_callback = None

    _ = pipe.fit(X, y=y)
    print(pipe.transform(X))
    # [[1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]
    #  [1111. 1111. 1111. 1111. 1111.]]

    # Logging also works with FeatureUnion
    from sklearn.pipeline import FeatureUnion
    pipe_w_default_log_callback = DebugPipeline(steps, log_callback="default")
    pipe_w_custom_log_callback = DebugPipeline(steps, log_callback=log_callback)

    pipe_union = DebugPipeline([
        ("feature_union", FeatureUnion([
            ("pipe_w_default_log_callback", pipe_w_default_log_callback),
            ("pipe_w_custom_log_callback", pipe_w_custom_log_callback),
        ])),
        ("final_adder", Adder(10000))
    ], log_callback="default")

    _ = pipe_union.fit(X, y=y)   # doctest:+ELLIPSIS
    # [default_log_callback:34] - [Adder(value=1)] shape=(3, 5) time=0s
    # [default_log_callback:34] - [Adder(value=10)] shape=(3, 5) time=0s
    # [default_log_callback:34] - [Adder(value=100)] shape=(3, 5) time=0s
    # [log_callback:20] - [Adder(value=1)] shape=(3, 5) nbytes=120 time=0s
    # [log_callback:20] - [Adder(value=10)] shape=(3, 5) nbytes=120 time=0s
    # [log_callback:20] - [Adder(value=100)] shape=(3, 5) nbytes=120 time=0s
    # [default_log_callback:34] - [FeatureUnion(...)] shape=(3, 10) time=0s

    print(pipe_union.transform(X))
    # [[11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111.]
    #  [11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111.]
    #  [11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111. 11111.]]
    ```
    """

    def __init__(self, steps, memory=None, verbose=False, *, log_callback=None):
        self.log_callback = log_callback
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    @property
    def memory(self):
        # When no log callback function is given, change nothing.
        # Or, if the memory cache was changed, set it back to its original.
        if self._log_callback is None:
            if hasattr(self._memory, "_cache"):
                self._memory.cache = self._memory._cache
            return self._memory

        self._memory = check_memory(self._memory)

        # Overwrite cache function of memory such that it logs the
        # output when the function is called
        if not hasattr(self._memory, "_cache"):
            self._memory._cache = self._memory.cache
        self._memory.cache = _cache_with_function_log_statement(self._log_callback).__get__(
            self._memory, self._memory.__class__
        )
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
        if self._log_callback == "default":
            self._log_callback = default_log_callback

    def __getstate__(self):
        """
        Prepare the state of the object for pickling.

        This method is called when the object is being pickled. It ensures that the `Memory` object is in its original
        state by temporarily restoring the original `cache` method and removing the custom `_cache` attribute. This is
        necessary because the custom `_cache` attribute is not picklable and would cause errors during the pickling
        process.

        Returns
        -------
        dict
            The state of the object to be pickled.
        """
        state = self.__dict__.copy()
        if hasattr(self._memory, "_cache"):
            self._memory.cache = self._memory._cache
            del self._memory._cache
        return state

    def __setstate__(self, state):
        """
        Restore the state of the object from the pickled state.

        This method is called when the object is being unpickled. It restores the state of the object and re-applies the
        custom `_cache` attribute by wrapping the `cache` method with the logging wrapper (`_cache_with_function_log_statement`).
        This ensures that the `Memory` object has the custom `_cache` attribute after unpickling.

        Parameters
        ----------
        state : dict
            The state of the object to be restored.
        """
        self.__dict__.update(state)
        if self._log_callback is not None:
            self._memory = check_memory(self._memory)
            if not hasattr(self._memory, "_cache"):
                self._memory._cache = self._memory.cache
            self._memory.cache = _cache_with_function_log_statement(self._log_callback).__get__(
                self._memory, self._memory.__class__
            )


def make_debug_pipeline(*steps, **kwargs):
    """Construct a `DebugPipeline` from the given estimators.

    This is a shorthand for the `DebugPipeline` constructor; it does not require, and does not permit, naming the
    estimators. Instead, their names will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list
        List of estimators to be included in the pipeline.
    **kwargs : dict
        Additional keyword arguments passed to the `DebugPipeline` constructor.
        Possible arguments are `memory`, `verbose` and `log_callback`:

        - `memory` : str | object with the joblib.Memory interface, default=None

            Used to cache the fitted transformers of the pipeline. The last step will never be cached, even if it is a
            transformer. By default, no caching is performed. If a string is given, it is the path to the caching
            directory. Enabling caching triggers a clone of the transformers before fitting. Therefore, the transformer
            instance given to the pipeline cannot be inspected directly. Use the attribute `named_steps` or `steps` to
            inspect estimators within the pipeline. Caching the transformers is advantageous when fitting is time
            consuming.

        - `verbose` : bool, default=False

            If True, the time elapsed while fitting each step will be printed as it is completed.

        - `log_callback` : str | Callable | None, default=None.

            The callback function that logs information in between each intermediate step. If set to `"default"`,
            `default_log_callback` is used.

    Returns
    -------
    DebugPipeline
        Instance with given steps, `memory`, `verbose` and `log_callback`.

    Examples
    --------
    ```py
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler

    make_debug_pipeline(StandardScaler(), GaussianNB(priors=None))
    # DebugPipeline(steps=[("standardscaler", StandardScaler()),
    #                 ("gaussiannb", GaussianNB())])
    ```

    See Also
    --------
    sklego.pipeline.DebugPipeline : Class for creating a pipeline of transforms with a final estimator.
    """
    memory = kwargs.pop("memory", None)
    verbose = kwargs.pop("verbose", False)
    log_callback = kwargs.pop("log_callback", None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0]))
    return DebugPipeline(
        _name_estimators(steps),
        memory=memory,
        verbose=verbose,
        log_callback=log_callback,
    )
