from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


def _identity(X):
    return X


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    The identity transformer returns what it is fed. Does not apply anything useful.
    The reason for having it is because you can build more expressive pipelines.
    """

    def fit(self, X, y=None):
        """'Fits' the estimator."""
        X = check_array(X, copy=True, estimator=self)
        self.fitted_ = True
        self.shape_ = X.shape
        return self

    def transform(self, X):
        """'Applies' the estimator."""
        X = check_array(X, copy=True, estimator=self, )
        check_is_fitted(self, 'fitted_', 'shape_')
        if X.shape[1] != self.shape_[1]:
            raise ValueError(f"Wrong shape is passed to transform. Trained on {self.shape_[1]} cols got {X.shape[1]}")
        return X
