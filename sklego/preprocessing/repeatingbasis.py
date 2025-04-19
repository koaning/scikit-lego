import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


class RepeatingBasisFunction(TransformerMixin, BaseEstimator):
    """The `RepeatingBasisFunction` transformer is designed to be used when the input data has a circular nature.

    For example, for days of the week you might face the problem that, conceptually, day 7 is as close to day 6 as it is
    to day 1. While numerically their distance is different.

    This transformer remedies that problem. The transformer selects a column and transforms it with a given number of
    repeating (radial) basis functions.

    Radial basis functions are bell-curve shaped functions which take the original data as input. The basis functions
    are equally spaced over the input range. The key feature of repeating basis functions is that they are continuous
    when moving from the max to the min of the input range. As a result these repeating basis functions can capture how
    close each datapoint is to the center of each repeating basis function, even when the input data has a circular
    nature.

    Parameters
    ----------
    column : int | str, default=0
        Index or column name of the data to transform. Integers are interpreted as positional columns, while
        strings can reference DataFrame columns by name.
    remainder : Literal["drop", "passthrough"], default="drop"
        By default, only the specified column is transformed, and the non-specified columns are dropped.
        By specifying `remainder="passthrough"`, all remaining columns will be automatically passed through.
        This subset of columns is concatenated with the output of the transformer.
    n_periods : int, default=12
        Number of basis functions to create, i.e., the number of columns that will exit the transformer.
    input_range : Tuple[float, float] | List[float] | None, default=None
        The values at which the data repeats itself. For example, for days of the week this is (1,7).
        If `input_range=None` it is inferred from the training data.
    width : float, default=1.0.
        Determines the width of the radial basis functions.

    Attributes
    ----------
    pipeline_ : ColumnTransformer
        Fitted `ColumnTransformer` object used to transform data with repeating basis functions.

    Examples
    --------
    ```py
    import pandas as pd
    from sklego.preprocessing import RepeatingBasisFunction

    df = pd.DataFrame({
        "user_id": [101, 102, 103],
        "created_day": [5, 1, 7]
    })
    RepeatingBasisFunction(column="created_day", input_range=(1,7)).fit_transform(df)
    # array([[0.06217652, 0.00432024, 0.16901332, 0.89483932, 0.64118039],
    #        [1.        , 0.36787944, 0.01831564, 0.01831564, 0.36787944],
    #        [1.        , 0.36787944, 0.01831564, 0.01831564, 0.36787944]])
    ```
    """

    def __init__(self, column=0, remainder="drop", n_periods=12, input_range=None, width=1.0):
        self.column = column
        self.remainder = remainder
        self.n_periods = n_periods
        self.input_range = input_range
        self.width = width

    def fit(self, X, y=None):
        """Fit `RepeatingBasisFunction` transformer on input data `X`.
        It uses `sklearn.compose.ColumnTransformer`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the quantiles for capping.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : RepeatingBasisFunction
            The fitted transformer.
        """
        self.pipeline_ = ColumnTransformer(
            [
                (
                    "repeatingbasis",
                    _RepeatingBasisFunction(
                        n_periods=self.n_periods,
                        input_range=self.input_range,
                        width=self.width,
                    ),
                    [self.column],
                )
            ],
            remainder=self.remainder,
        )

        self.pipeline_.fit(X, y)

        return self

    def transform(self, X):
        """Transform input data `X` with fitted `RepeatingBasisFunction` transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_periods)
            Transformed data.
        """
        check_is_fitted(self, ["pipeline_"])
        return self.pipeline_.transform(X)


class _RepeatingBasisFunction(TransformerMixin, BaseEstimator):
    """Transformer for generating repeating basis functions.

    This transformer generates a set of repeating basis functions for a given input data. Each basis function is
    defined by its center, and the width of the functions is adjusted based on the number of periods. It is
    particularly useful in applications where periodic patterns need to be captured.

    Parameters
    ----------
    n_periods : int, default=12
        The number of repeating periods or basis functions to generate.
    input_range : Tuple[float, float] | List[float] | None, default=None
        The values at which the data repeats itself. For example, for days of the week this is (1,7).
        If `input_range=None` it is inferred from the training data.
    width : float, default=1.0
        The width of the basis functions. This parameter controls how narrow or wide the basis functions are.

    Attributes
    ----------
    bases_ : numpy.ndarray of shape (n_periods,)
        The centers of the repeating basis functions.
    width_ : float
        The adjusted width of the basis functions based on the number of periods and the provided width.
    """

    def __init__(self, n_periods: int = 12, input_range=None, width: float = 1.0):
        self.n_periods = n_periods
        self.input_range = input_range
        self.width = width

    def fit(self, X, y=None):
        """Fit the transformer to the input data and compute the basis functions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data used to compute the basis functions.
        y : array-like of shape (n_samples), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : _RepeatingBasisFunction
            The fitted transformer.
        """
        X = validate_data(self, X=X, ensure_2d=True, reset=True)

        # find min and max for standardization if not given explicitly
        if self.input_range is None:
            self.input_range = (X.min(), X.max())

        # exclude the last value because it's identical to the first for repeating basis functions
        self.bases_ = np.linspace(0, 1, self.n_periods + 1)[:-1]

        # curves should narrower (wider) when we have more (fewer) basis functions
        self.width_ = self.width / self.n_periods

        return self

    def transform(self, X):
        """Transform the input data into features based on the repeating basis functions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be transformed using the basis functions.

        Returns
        -------
        array-like of shape (n_samples, n_periods)
            The transformed data with features generated from the basis functions.

        Raises
        ------
        ValueError
            If X has more than one column, as this transformer only accepts one feature as input.
        """
        check_is_fitted(self, ["bases_", "width_"])
        X = validate_data(self, X=X, ensure_2d=True, reset=False)

        # MinMax Scale to 0-1
        X = (X - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        base_distances = self._array_bases_distances(X, self.bases_)

        # apply rbf function to series for each basis
        return self._rbf(base_distances)

    def _array_base_distance(self, arr: np.ndarray, base: float) -> np.ndarray:
        """Calculate the distances between all array values and the base, where 0 and 1 are assumed to be at the same
        positions

        Parameters
        ----------
        arr : np.ndarray, shape (n_samples,)
            The input array for which distances to the base are calculated.
        base : float
            The base value to which distances are calculated.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            An array of distances between the values in `arr` and the `base`, with consideration of 0 and 1 as
            equivalent positions.
        """
        abs_diff_0 = np.abs(arr - base)
        abs_diff_1 = 1 - abs_diff_0
        concat = np.concatenate((abs_diff_0.reshape(-1, 1), abs_diff_1.reshape(-1, 1)), axis=1)
        final = concat.min(axis=1)
        return final

    def _array_bases_distances(self, array, bases):
        """Calculate distances between all combinations of array and bases values.

        Parameters
        ----------
        array : np.ndarray, shape (n_samples,)
            The input array for which distances to the bases are calculated.
        bases : np.ndarray, shape (n_bases,)
            The bases values to which distances are calculated.

        Returns
        -------
        np.ndarray, shape (n_samples, n_bases)
            An array of distances between the elements of 'array' and the specified 'bases'.
        """
        array = array.reshape(-1, 1)
        bases = bases.reshape(1, -1)

        return np.apply_along_axis(lambda b: self._array_base_distance(array, base=b), axis=0, arr=bases)

    def _rbf(self, arr):
        """Apply the Radial Basis Function (RBF) to the input array.

        Parameters
        ----------
        arr : np.ndarray
            The input array to which the RBF is applied.

        Returns
        -------
        np.ndarray
            An array with the RBF applied to the input array.
        """
        return np.exp(-((arr / self.width_) ** 2))
