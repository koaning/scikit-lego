import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class BayesianGMMClassifier(ClassifierMixin, BaseEstimator):
    """The `BayesianGMMClassifier` trains a Gaussian Mixture Model for each class in `y` on a dataset `X`.
    Once a density is trained for each class we can evaluate the likelihood scores to see which class is more likely.

    !!! note
        All the parameters are an exact copy of those of
        [sklearn.mixture.BayesianGaussianMixture]( https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html).


    Attributes
    ----------
    gmms_ : dict[int, BayesianGaussianMixture]
        A dictionary of Bayesian Gaussian Mixture Models, one for each class.
    classes_ : np.ndarray of shape (n_classes,)
        The classes seen during `fit`.

    Examples
    --------

    ```python
    import numpy as np
    from sklego.mixture import BayesianGMMClassifier

    # Generate datset
    np.random.seed(1)
    group0 = np.random.normal(0, 3, (1000, 2))
    group1 = np.random.normal(2.5, 2, (500, 2))
    data = np.vstack([group0, group1])

    y = np.hstack([np.zeros((group0.shape[0],), dtype=int), np.ones((group1.shape[0],), dtype=int)])
    # Create and fit the BayesianGMMClassifier model
    bgmm = BayesianGMMClassifier(n_components=2, random_state=1)
    bgmm.fit(data, y)

    # Classify the train dataset into two clusters (n_components=2)
    labels = bgmm.predict(data)

    # Classify a new point into one of two clusters
    p = np.array([[1.5, 0.5]])
    p_prob = bgmm.predict_proba(p) # predict the probabilities p belongs to each cluster
    print(f'Probability point p belongs to group1 is {p_prob[0,0]:.2f}')
    ### Probability point p belongs to group1 is 0.38
    print(f'Probability point p belongs to group2 is {p_prob[0,1]:.2f}')
    ### Probability point p belongs to group2 is 0.62

    print(f'It is more probable that point p belongs to group{np.argmax(p_prob)}')
    ### It is more probable that point p belongs to group1
    ```
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=0.001,
        reg_covar=1e-06,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianGMMClassifier":
        """Fit the `BayesianGMMClassifier` model using `X`, `y` as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : BayesianGMMClassifier
            The fitted estimator.
        """
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            mixture = BayesianGaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                tol=self.tol,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                init_params=self.init_params,
                weight_concentration_prior_type=self.weight_concentration_prior_type,
                weight_concentration_prior=self.weight_concentration_prior,
                mean_precision_prior=self.mean_precision_prior,
                mean_prior=self.mean_prior,
                degrees_of_freedom_prior=self.degrees_of_freedom_prior,
                covariance_prior=self.covariance_prior,
                random_state=self.random_state,
                warm_start=self.warm_start,
                verbose=self.verbose,
                verbose_interval=self.verbose_interval,
            )
            self.gmms_[c] = mixture.fit(subset_x, subset_y)

        self.n_iter_ = sum(mixture.n_iter_ for mixture in self.gmms_.values())

        return self

    def predict(self, X):
        """Predict labels for `X` using fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        check_is_fitted(self, ["gmms_", "classes_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        """Predict probabilities for `X` using fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        check_is_fitted(self, ["gmms_", "classes_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        res = np.zeros((X.shape[0], self.classes_.shape[0]))
        for idx, c in enumerate(self.classes_):
            res[:, idx] = self.gmms_[c].score_samples(X)
        return softmax(res, axis=1)
