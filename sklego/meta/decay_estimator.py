from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import _check_n_features, validate_data

from sklego.meta._decay_utils import exponential_decay, linear_decay, sigmoid_decay, stepwise_decay


class DecayEstimator(MetaEstimatorMixin, BaseEstimator):
    """Morphs an estimator such that the training weights can be adapted to ensure that points that are far away have
    less weight.

    This meta estimator will only work for estimators that allow a `sample_weights` argument in their `.fit()` method.
    The meta estimator `.fit()` method computes the weights to pass to the estimator's `.fit()` method.

    !!! warning
        It is up to the user to sort the dataset appropriately.

    !!! warning
        By default all the checks on the inputs `X` and `y` are delegated to the wrapped estimator.
        To change such behaviour, set `check_input` to `True`.
        Remark that if the check is skipped, then `y` should have a `shape` attribute, which is
        used to extract the number of samples in training data, and compute the weights.

    Parameters
    ----------
    model : scikit-learn compatible estimator
        The estimator to be wrapped.
    decay_func : Literal["linear", "exponential", "stepwise", "sigmoid"] | \
            Callable[[np.ndarray, np.ndarray, ...], np.ndarray], default="exponential"
        The decay function to use. Available built-in decay functions are:

        - `"linear"`: linear decay from `max_value` to `min_value`.
        - `"exponential"`: exponential decay with decay rate `decay_rate`.
        - `"stepwise"`: stepwise decay from `max_value` to `min_value`, with `n_steps` steps or step size `step_size`.
        - `"sigmoid"`: sigmoid decay from `max_value` to `min_value` with decay rate `growth_rate`.

        Otherwise a callable can be passed and it should accept `X`, `y` as first two positional arguments and any other
        keyword argument passed along from `decay_kwargs` (if any). It should compute the weights and return an array
        of shape `(n_samples,)`.
    check_input : bool, default=False
        Whether or not to check the input data. If False, the checks are delegated to the wrapped estimator.
    decay_kwargs : dict | None, default=None
        Keyword arguments to the decay function.

    Attributes
    ----------
    estimator_ : scikit-learn compatible estimator
        The fitted estimator.
    weights_ : array-like of shape (n_samples,)
        The weights used to train the estimator.
    classes_ : array-like of shape (n_classes,)
        The classes labels. Only present if the wrapped estimator is a classifier.

    Example
    --------
    ```py
    from sklearn.linear_model import LinearRegression
    from sklego.meta import DecayEstimator

    decay_estimator = DecayEstimator(
        model=LinearRegression(),
        decay_func="linear",
        decay_kwargs={"min_value":0.1, "max_value":0.9}
        )

    X, y = ...

    # Fit the DecayEstimator on the data, this will compute the weights
    # and pass them to the wrapped estimator
    _ = decay_estimator.fit(X, y)

    # At prediction time, the weights are not used
    predictions = decay_estimator.predict(X)

    # The weights are stored in the `weights_` attribute
    weights = decay_estimator.weights_
    ```
    """

    _ALLOWED_DECAYS = {
        "linear": linear_decay,
        "exponential": exponential_decay,
        "stepwise": stepwise_decay,
        "sigmoid": sigmoid_decay,
    }

    _required_parameters = ["model"]

    def __init__(self, model, decay_func="exponential", check_input=False, decay_kwargs=None):
        self.model = model
        self.decay_func = decay_func
        self.check_input = check_input
        self.decay_kwargs = decay_kwargs

    def _is_classifier(self):
        """Checks if the wrapped estimator is a classifier."""
        return any(["ClassifierMixin" in p.__name__ for p in type(self.model).__bases__])

    def _is_regressor(self):
        """Checks if the wrapped estimator is a regressor."""
        return any(["RegressorMixin" in p.__name__ for p in type(self.model).__bases__])

    @property
    def _estimator_type(self):
        """Computes `_estimator_type` dynamically from the wrapped model."""
        return self.model._estimator_type

    def fit(self, X, y):
        """Fit the underlying estimator on the training data `X` and `y` using the calculated sample weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DecayEstimator
            The fitted estimator.
        """

        if self.check_input:
            X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, accept_sparse=True, reset=True)
        else:
            _check_n_features(self, X, reset=True)

        if self.decay_func in self._ALLOWED_DECAYS.keys():
            self.decay_func_ = self._ALLOWED_DECAYS[self.decay_func]
        elif callable(self.decay_func):
            self.decay_func_ = self.decay_func
        else:
            raise ValueError(f"`decay_func` should be one of {self._ALLOWED_DECAYS.keys()} or a callable")

        self.weights_ = self.decay_func_(X, y, **(self.decay_kwargs or {}))
        self.estimator_ = clone(self.model)

        try:
            self.estimator_.fit(X, y, sample_weight=self.weights_)
        except TypeError as e:
            if "sample_weight" in str(e):
                raise TypeError(f"Model {type(self.model).__name__}.fit() does not have 'sample_weight'")

        if self._is_classifier():
            self.classes_ = self.estimator_.classes_

        return self

    def predict(self, X):
        """Predict target values for `X` using trained underlying estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        if self._is_classifier():
            check_is_fitted(self, ["classes_"])

        check_is_fitted(self, ["weights_", "estimator_"])
        return self.estimator_.predict(X)

    def score(self, X, y):
        """Alias for `.score()` method of the underlying estimator."""
        return self.estimator_.score(X, y)

    def __sklearn_tags__(self):
        return self.model.__sklearn_tags__()
