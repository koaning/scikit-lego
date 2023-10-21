from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

######################################## DebugPipeline ###########################################
##########################################################################################

# --8<-- [start:setup]
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklego.pipeline import DebugPipeline

logging.basicConfig(
    format=("[%(funcName)s:%(lineno)d] - %(message)s"),
    level=logging.INFO
)
# --8<-- [end:setup]

# --8<-- [start:simple-pipe]
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
# --8<-- [end:simple-pipe]

# --8<-- [start:simple-pipe-fit-transform]
pipe = DebugPipeline(steps)
_ = pipe.fit(X, y=y)

X_out = pipe.transform(X)
print("Transformed X:\n", X_out)
# --8<-- [end:simple-pipe-fit-transform]

# --8<-- [start:log-callback]
pipe = DebugPipeline(steps, log_callback="default")
_ = pipe.fit(X, y=y)

X_out = pipe.transform(X)
print("Transformed X:\n", X_out)
# --8<-- [end:log-callback]

# --8<-- [start:log-callback-after]
pipe = DebugPipeline(steps)
pipe.log_callback = "default"

_ = pipe.fit(X, y=y)

X_out = pipe.transform(X)
print("Transformed X:\n", X_out)
# --8<-- [end:log-callback-after]

# --8<-- [start:custom-log-callback]
def log_callback(output, execution_time, **kwargs):
    """My custom `log_callback` function

    Parameters
    ----------
    output : tuple(
            numpy.ndarray or pandas.DataFrame
            :class:estimator or :class:transformer
        )
        The output of the step and a step in the pipeline.
    execution_time : float
        The execution time of the step.
    """
    logger = logging.getLogger(__name__)
    step_result, step = output
    logger.info(f"[{step}] shape={step_result.shape} "
                f"nbytes={step_result.nbytes} time={execution_time}")


pipe.log_callback = log_callback
_ = pipe.fit(X, y=y)

X_out = pipe.transform(X)
print("Transformed X:\n", X_out)
# --8<-- [end:custom-log-callback]


# --8<-- [start:feature-union]
from sklearn.pipeline import FeatureUnion

pipe_w_default_log_callback = DebugPipeline(steps, log_callback='default')
pipe_w_custom_log_callback = DebugPipeline(steps, log_callback=log_callback)

pipe_union = FeatureUnion([
    ('pipe_w_default_log_callback', pipe_w_default_log_callback),
    ('pipe_w_custom_log_callback', pipe_w_custom_log_callback),
])

_ = pipe_union.fit(X, y=y)

X_out = pipe_union.transform(X)
print('Transformed X:\n', X_out)
# --8<-- [end:feature-union]


# --8<-- [start:remove]
pipe.log_callback = None
_ = pipe.fit(X, y=y)

X_out = pipe.transform(X)
print('Transformed X:\n', X_out)
# --8<-- [end:remove]
