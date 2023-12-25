from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """The `IdentityTransformer` returns what it is fed. Does not apply any transformation.

    The reason for having it is because you can build more expressive pipelines.

    Parameters
    ----------
    check_X : bool, default=False
        Whether to validate `X` to be non-empty 2D array of finite values and attempt to cast `X` to float.
        If disabled, the model/pipeline is expected to handle e.g. missing, non-numeric, or non-finite values.

    Attributes
    ----------
    n_samples_ : int
        The number of samples seen during `fit`.
    n_features_in_ : int
        The number of features seen during `fit`.
    shape_ : tuple[int, int]
        Deprecated, please use `n_samples_` and `n_features_in_` instead.
    """

    def __init__(self, check_X: bool = False):
        self.check_X = check_X

    def fit(self, X, y=None):
        """Check the input data if `check_X` is enabled and and records its shape.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : IdentityTransformer
            The fitted transformer.
        """
        if self.check_X:
            X = check_array(X, copy=True, estimator=self)
        self.n_samples_, self.n_features_in_ = X.shape
        return self

    def transform(self, X):
        """Performs identity "transformation" on `X` - which is no transformation at all.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            Unchanged input data.

        Raises
        ------
        ValueError
            If the number of columns from `X` differs from the number of columns when fitting.
        """
        if self.check_X:
            X = check_array(X, copy=True, estimator=self)
        check_is_fitted(self, "n_features_in_")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Wrong shape is passed to transform. Trained on {self.n_features_in_} cols got {X.shape[1]}"
            )
        return X

    @property
    def shape_(self):
        """Returns the shape of the estimator."""
        return (self.n_samples_, self.n_features_in_)
