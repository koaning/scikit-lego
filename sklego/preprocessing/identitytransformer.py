from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    The identity transformer returns what it is fed. Does not apply anything useful.
    The reason for having it is because you can build more expressive pipelines.

    :type check_X: bool, optional, default=False
    :param check_X: Whether to validate X to be non-empty 2D array of finite values
                    and attempt to cast X to float.
                    If disabled, the model/pipeline is expected to handle e.g. missing,
                    non-numeric, or non-finite values.
    """
    def __init__(
        self,
        check_X: bool = False
    ):
        self.check_X = check_X

    def fit(self, X, y=None):
        """'Fits' the estimator."""
        if self.check_X:
            X = check_array(X, copy=True, estimator=self)
        self.shape_ = X.shape
        return self

    def transform(self, X):
        """'Applies' the estimator."""
        if self.check_X:
            X = check_array(X, copy=True, estimator=self)
        check_is_fitted(self, 'shape_')
        if X.shape[1] != self.shape_[1]:
            raise ValueError(f"Wrong shape is passed to transform. Trained on {self.shape_[1]} cols got {X.shape[1]}")
        return X
