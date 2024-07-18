from warnings import warn

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_random_state

from sklego.common import TrainOnlyTransformerMixin


class RandomAdder(TrainOnlyTransformerMixin, BaseEstimator):
    """The `RandomAdder` transformer adds random noise to the input data.

    This class is designed to be used during the training phase and not for transforming test data.
    Noise added is sampled from a normal distribution with mean 0 and standard deviation `noise`.

    Parameters
    ----------
    noise : float, default=1.0
        The standard deviation of the normal distribution from which the noise is sampled.
    random_state : int | None
        The seed used by the random number generator.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during `fit`.
    dim_ : int
        Deprecated, please use `n_features_in_` instead.

    Examples
    --------
    ```py
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklego.preprocessing import RandomAdder

    # Create a pipeline with the RandomAdder and a LinearRegression model
    pipeline = Pipeline([
        ('random_adder', RandomAdder(noise=0.5, random_state=42)),
        ('linear_regression', LinearRegression())
    ])

    # Fit the pipeline with training data
    pipeline.fit(X_train, y_train)

    # Use the fitted pipeline to make predictions
    y_pred = pipeline.predict(X_test)
    ```
    """

    def __init__(self, noise=1, random_state=None):
        self.noise = noise
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the transformer on training data `X` and `y` by checking the input data and record the number of
        input features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomAdder
            The fitted transformer.
        """
        super().fit(X, y)
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.n_features_in_ = X.shape[1]

        return self

    def transform_train(self, X):
        r"""Transform training data by adding random noise sampled from $N(0, \text{noise})$.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data for which the noise will be added.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            The data with the noise added.
        """
        rs = check_random_state(self.random_state)
        check_is_fitted(self, ["n_features_in_"])

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        return X + rs.normal(0, self.noise, size=X.shape)

    @property
    def dim_(self):
        warn(
            "Please use `n_features_in_` instead of `dim_`, `dim_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_

    def _more_tags(self):
        return {"non_deterministic": True}
