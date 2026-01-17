import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import _check_n_features, validate_data


class LinearEmbedder(TransformerMixin, BaseEstimator):
    """Embed features using coefficients from a linear model.

    The LinearEmbedder fits a linear model to the training data and uses the
    learned coefficients to rescale the features. This can improve the
    representation of the features for downstream models, particularly KNN.

    Parameters
    ----------
    estimator : sklearn estimator, default=Ridge(fit_intercept=False)
        The linear estimator to use for learning the coefficients.
    check_input : bool, default=True
        Whether to validate input data.

    Attributes
    ----------
    estimator_ : sklearn estimator
        The fitted linear estimator.
    coef_ : ndarray of shape (n_features,) or (n_features, n_targets)
        The learned coefficients used for embedding.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    ```py
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklego.preprocessing import LinearEmbedder

    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Use LinearEmbedder to improve KNN performance
    pipe = Pipeline([
        ('embedder', LinearEmbedder()),
        ('knn', KNeighborsRegressor(n_neighbors=5))
    ])

    pipe.fit(X, y)
    predictions = pipe.predict(X)
    ```
    """

    def __init__(self, estimator=None, check_input=True):
        self.estimator = estimator
        self.check_input = check_input

    def fit(self, X, y):
        """Fit the linear estimator and store the coefficients for embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : LinearEmbedder
            The fitted embedder.
        """
        if self.check_input:
            X, y = validate_data(self, X=X, y=y, reset=True, multi_output=True)
        else:
            _check_n_features(self, X, reset=True)

        # Use Ridge with fit_intercept=False as default
        if self.estimator is None:
            self.estimator_ = Ridge(fit_intercept=False)
        else:
            self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y)

        # Store coefficients for embedding
        self.coef_ = self.estimator_.coef_

        # Handle different coefficient shapes
        if len(self.coef_.shape) == 1:
            self.coef_ = self.coef_.reshape(1, -1)
        elif len(self.coef_.shape) == 2 and self.coef_.shape[0] == 1:
            # Keep as is for single output
            pass
        else:
            # For multi-output, take the mean across outputs
            self.coef_ = np.mean(self.coef_, axis=0).reshape(1, -1)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Transform the data by rescaling with learned coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Rescaled data using the learned coefficients.
        """
        check_is_fitted(self, "coef_")

        if self.check_input:
            X = validate_data(self, X=X, reset=False)
        else:
            _check_n_features(self, X, reset=False)

        # Apply coefficient scaling
        X_transformed = X * self.coef_

        return X_transformed
