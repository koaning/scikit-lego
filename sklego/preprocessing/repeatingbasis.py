import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class RepeatingBasisFunction(TransformerMixin, BaseEstimator):
    """
    This is a transformer for features with some form of circularity.
    E.g. for days of the week you might face the problem that, conceptually, day 7 is as
    close to day 6 as it is to day 1. While numerically their distance is different.
    This transformer remedies that problem.
    The transformer selects a column and transforms it with a given number of repeating
    (radial) basis functions. Radial basis functions are bell-curve shaped functions
    which take the original data as input. The basis functions are equally spaced over
    the input range. The key feature of repeating basis functions is that they are
    continuous when moving from the max to the min of the input range. As a result these
    repeating basis functions can capture how close each datapoint is to the center of
    each repeating basis function, even when the input data has a circular nature.

    :type column: int or list, default=0
    :param column: Indexes the data on its second axis. Integers are interpreted as
        positional columns, while strings can reference DataFrame columns by name.

    :type remainder: {'drop', 'passthrough'}, default="drop"
    :param remainder: By default, only the specified column is transformed, and the
        non-specified columns are dropped. (default of ``'drop'``). By specifying
        ``remainder='passthrough'``, all remaining columns will be automatically passed
        through. This subset of columns is concatenated with the output of the transformer.

    :type n_periods: int, default=12
    :param n_periods: number of basis functions to create, i.e., the number of columns that
        will exit the transformer.

    :type input_range: tuple or None, default=None
    :param input_range: the values at which the data repeats itself. For example, for days of
        the week this is (1,7). If input_range=None it is inferred from the training data.
    """

    def __init__(self, column=0, remainder="drop", n_periods=12, input_range=None):
        self.column = column
        self.remainder = remainder
        self.n_periods = n_periods
        self.input_range = input_range

    def fit(self, X, y=None):
        self.pipeline_ = ColumnTransformer(
            [
                (
                    "repeatingbasis",
                    _RepeatingBasisFunction(
                        n_periods=self.n_periods, input_range=self.input_range
                    ),
                    [self.column],
                )
            ],
            remainder=self.remainder,
        )

        self.pipeline_.fit(X, y)

        return self

    def transform(self, X):
        check_is_fitted(self, ["pipeline_"])
        return self.pipeline_.transform(X)


class _RepeatingBasisFunction(TransformerMixin, BaseEstimator):
    def __init__(self, n_periods: int = 12, input_range=None):
        self.n_periods = n_periods
        self.input_range = input_range

    def fit(self, X, y=None):
        X = check_array(X, estimator=self)

        # find min and max for standardization if not given explicitly
        if self.input_range is None:
            self.input_range = (X.min(), X.max())

        # exclude the last value because it's identical to the first for repeating basis functions
        self.bases_ = np.linspace(0, 1, self.n_periods + 1)[:-1]

        # curves should narrower (wider) when we have more (fewer) basis functions
        self.width_ = 1 / self.n_periods

        return self

    def transform(self, X):
        X = check_array(X, estimator=self, ensure_2d=True)
        check_is_fitted(self, ["bases_", "width_"])
        # This transformer only accepts one feature as input
        if X.shape[1] != 1:
            raise ValueError(f"X should have exactly one column, it has: {X.shape[1]}")

        # MinMax Scale to 0-1
        X = (X - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        base_distances = self._array_bases_distances(X, self.bases_)

        # apply rbf function to series for each basis
        return self._rbf(base_distances)

    def _array_base_distance(self, arr: np.ndarray, base: float) -> np.ndarray:
        """Calculates the distances between all array values and the base,
        where 0 and 1 are assumed to be at the same position"""
        abs_diff_0 = np.abs(arr - base)
        abs_diff_1 = 1 - abs_diff_0
        concat = np.concatenate(
            (abs_diff_0.reshape(-1, 1), abs_diff_1.reshape(-1, 1)), axis=1
        )
        final = concat.min(axis=1)
        return final

    def _array_bases_distances(self, array, bases):
        """Calculates the distances between all combinations of array and bases values"""
        array = array.reshape(-1, 1)
        bases = bases.reshape(1, -1)

        return np.apply_along_axis(
            lambda b: self._array_base_distance(array, base=b), axis=0, arr=bases
        )

    def _rbf(self, arr):
        return np.exp(-((arr / self.width_) ** 2))
