import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    MetaEstimatorMixin,
)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    check_array,
    FLOAT_DTYPES,
)

from sklego.base import ProbabilisticClassifier
from sklego.common import as_list, expanding_list, TrainOnlyTransformerMixin


class EstimatorTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """
    Allows using an estimator such as a model as a transformer in an earlier step of a pipeline

    :param estimator: An instance of the estimator that should be used for the transformation
    :param predict_func: The function called on the estimator when transforming e.g. (`predict`, `predict_proba`)
    """

    def __init__(self, estimator, predict_func="predict"):
        self.estimator = estimator
        self.predict_func = predict_func

    def fit(self, X, y):
        """Fits the estimator"""
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        """
        Applies the `predict_func` on the fitted estimator.

        Returns an array of shape `(X.shape[0], )`.
        """
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, self.predict_func)(X).reshape(-1, 1)


def constant_shrinkage(group_sizes: list, alpha: float) -> np.ndarray:
    r"""
    The augmented prediction for each level is the weighted average between its prediction and the augmented
    prediction for its parent.

    Let $\hat{y}_i$ be the prediction at level $i$, with $i=0$ being the root, than the augmented prediction
    $\hat{y}_i^* = \alpha \hat{y}_i + (1 - \alpha) \hat{y}_{i-1}^*$, with $\hat{y}_0^* = \hat{y}_0$.
    """
    return np.array(
        [alpha ** (len(group_sizes) - 1)]
        + [
            alpha ** (len(group_sizes) - 1 - i) * (1 - alpha)
            for i in range(1, len(group_sizes) - 1)
        ]
        + [(1 - alpha)]
    )


def relative_shrinkage(group_sizes: list) -> np.ndarray:
    """Weigh each group according to it's size"""
    return np.array(group_sizes)


def min_n_obs_shrinkage(group_sizes: list, min_n_obs) -> np.ndarray:
    """Use only the smallest group with a certain amount of observations"""
    if min_n_obs > max(group_sizes):
        raise ValueError(
            f"There is no group with size greater than or equal to {min_n_obs}"
        )

    res = np.zeros(len(group_sizes))
    res[np.argmin(np.array(group_sizes) >= min_n_obs) - 1] = 1
    return res


class GroupedEstimator(BaseEstimator):
    """
    Construct an estimator per data group. Splits data by values of a
    single column and fits one estimator per such column.

    :param estimator: the model/pipeline to be applied per group
    :param groups: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param value_columns: Columns to use in the prediction. If None (default), use all non-grouping columns
    :param shrinkage: How to perform shrinkage.
                      None: No shrinkage (default)
                      {"constant", "min_n_obs", "relative"} or a callable
                      * constant: shrunk prediction for a level is weighted average of its prediction and its
                                  parents prediction
                      * min_n_obs: shrunk prediction is the prediction for the smallest group with at least
                                   n observations in it
                      * relative: each group-level is weight according to its size
                      * function: a function that takes a list of group lengths and returns an array of the
                                  same size with the weights for each group
    :param use_global_model: With shrinkage: whether to have a model over the entire input as first group
                             Without shrinkage: whether or not to fall back to a general model in case the group
                             parameter is not found during `.predict()`
    :param **shrinkage_kwargs: keyword arguments to the shrinkage function
    """

    def __init__(
        self,
        estimator,
        groups,
        value_columns=None,
        shrinkage=None,
        use_global_model=True,
        **shrinkage_kwargs,
    ):
        self.estimator = estimator
        self.groups = groups
        self.value_columns = value_columns
        self.shrinkage = shrinkage
        self.use_global_model = use_global_model
        self.shrinkage_kwargs = shrinkage_kwargs

    def __set_shrinkage_function(self):
        if isinstance(self.shrinkage, str):
            # Predefined shrinkage functions
            shrink_options = {
                "constant": constant_shrinkage,
                "relative": relative_shrinkage,
                "min_n_obs": min_n_obs_shrinkage,
            }

            try:
                self.shrinkage_function_ = shrink_options[self.shrinkage]
            except KeyError:
                raise ValueError(
                    f"The specified shrinkage function {self.shrinkage} is not valid, "
                    f"choose from {list(shrink_options.keys())} or supply a callable."
                )
        elif callable(self.shrinkage):
            self.__check_shrinkage_func()
            self.shrinkage_function_ = self.shrinkage
        else:
            raise ValueError(
                f"Invalid shrinkage specified. Should be either None (no shrinkage), str or callable."
            )

    def __check_shrinkage_func(self):
        """Validate the shrinkage function if a function is specified"""
        group_lengths = [10, 5, 2]
        expected_shape = np.array(group_lengths).shape
        try:
            result = self.shrinkage(group_lengths)
        except Exception as e:
            raise ValueError(
                f"Caught an exception while checking the shrinkage function: {str(e)}"
            ) from e
        else:
            if not isinstance(result, np.ndarray):
                raise ValueError(
                    f"shrinkage_function({group_lengths}) should return an np.ndarray"
                )
            if result.shape != expected_shape:
                raise ValueError(
                    f"shrinkage_function({group_lengths}).shape should be {expected_shape}"
                )

    @staticmethod
    def __check_cols_exist(X, cols):
        """Check whether the specified grouping columns are in X"""
        if X.shape[1] == 0:
            raise ValueError(
                f"0 feature(s) (shape=({X.shape[0]}, 0)) while a minimum of 1 is required."
            )

        # X has been converted to a DataFrame
        x_cols = set(X.columns)
        diff = set(as_list(cols)) - x_cols

        if len(diff) > 0:
            raise ValueError(f"{diff} not in columns of X {x_cols}")

    @staticmethod
    def __check_missing_and_inf(X):
        """Check that all elements of X are non-missing and finite, needed because check_array cannot handle strings"""
        if np.any(pd.isnull(X)):
            raise ValueError("X has NaN values")
        try:
            if np.any(np.isinf(X)):
                raise ValueError("X has infinite values")
        except TypeError:
            # if X cannot be converted to numeric, checking infinites does not make sense
            pass

    def __validate(self, X, y=None):
        """Validate the input, used in both fit and predict"""
        if (
            self.shrinkage
            and len(as_list(self.groups)) == 1
            and not self.use_global_model
        ):
            raise ValueError(
                "Cannot do shrinkage with a single group if use_global_model is False"
            )

        self.__check_cols_exist(X, self.value_colnames_)
        self.__check_cols_exist(X, self.group_colnames_)

        # Split the model data from the grouping columns, this part is checked `regularly`
        X_data = X.loc[:, self.value_colnames_]

        # y can be None because __validate used in predict, X can have no columns if the estimator only uses y
        if X_data.shape[1] > 0 and y is not None:
            check_X_y(X_data, y, multi_output=True)
        elif y is not None:
            check_array(y, ensure_2d=False)
        elif X_data.shape[1] > 0:
            check_array(X_data)

        self.__check_missing_and_inf(X)

    def __fit_grouped_estimator(self, X, y, value_columns, group_columns):
        # Reset indices such that they are the same in X and y
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        group_indices = X.groupby(group_columns).indices

        grouped_estimations = {
            group: clone(self.estimator).fit(
                X.loc[indices, value_columns], y.loc[indices]
            )
            for group, indices in group_indices.items()
        }

        return grouped_estimations

    def __get_shrinkage_factor(self, X):
        """Get for all complete groups an array of shrinkages"""
        counts = X.groupby(self.group_colnames_).size()

        # Groups that are split on all
        most_granular_groups = [
            grp
            for grp in self.groups_
            if len(as_list(grp)) == len(self.group_colnames_)
        ]

        # For each hierarchy level in each most granular group, get the number of observations
        hierarchical_counts = {
            granular_group: [
                counts[tuple(subgroup)].sum()
                for subgroup in expanding_list(granular_group, tuple)
            ]
            for granular_group in most_granular_groups
        }

        # For each hierarchy level in each most granular group, get the shrinkage factor
        shrinkage_factors = {
            group: self.shrinkage_function_(counts, **self.shrinkage_kwargs)
            for group, counts in hierarchical_counts.items()
        }

        # Make sure that the factors sum to one
        shrinkage_factors = {
            group: value / value.sum() for group, value in shrinkage_factors.items()
        }

        return shrinkage_factors

    def __prepare_input_data(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])

        if self.shrinkage is not None and self.use_global_model:
            global_col = "a-column-that-is-constant-for-all-data"
            X = X.assign(**{global_col: "global"})
            self.groups = [global_col] + as_list(self.groups)

        if y is not None:
            if isinstance(y, np.ndarray):
                pred_col = (
                    "the-column-that-i-want-to-predict-but-dont-have-the-name-for"
                )
                cols = (
                    pred_col
                    if y.ndim == 1
                    else ["_".join([pred_col, i]) for i in range(y.shape[1])]
                )
                y = (
                    pd.Series(y, name=cols)
                    if y.ndim == 1
                    else pd.DataFrame(y, columns=cols)
                )

            return X, y

        return X

    def fit(self, X, y=None):
        """
        Fit the model using X, y as training data. Will also learn the groups that exist within the dataset.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = self.__prepare_input_data(X, y)

        if self.shrinkage is not None:
            self.__set_shrinkage_function()

        self.group_colnames_ = [str(_) for _ in as_list(self.groups)]

        if self.value_columns is not None:
            self.value_colnames_ = [str(_) for _ in as_list(self.value_columns)]
        else:
            self.value_colnames_ = [
                _ for _ in X.columns if _ not in self.group_colnames_
            ]
        self.__validate(X, y)

        # List of all hierarchical subsets of columns
        self.group_colnames_hierarchical_ = expanding_list(self.group_colnames_, list)

        self.fallback_ = None

        if self.shrinkage is None and self.use_global_model:
            subset_x = X[self.value_colnames_]
            self.fallback_ = clone(self.estimator).fit(subset_x, y)

        if self.shrinkage is not None:
            self.estimators_ = {}

            for level_colnames in self.group_colnames_hierarchical_:
                self.estimators_.update(
                    self.__fit_grouped_estimator(
                        X, y, self.value_colnames_, level_colnames
                    )
                )
        else:
            self.estimators_ = self.__fit_grouped_estimator(
                X, y, self.value_colnames_, self.group_colnames_
            )

        self.groups_ = as_list(self.estimators_.keys())

        if self.shrinkage is not None:
            self.shrinkage_factors_ = self.__get_shrinkage_factor(X)

        return self

    def __predict_group(self, X, group_colnames):
        """Make predictions for all groups"""
        try:
            return (
                X.groupby(group_colnames, as_index=False)
                .apply(
                    lambda d: pd.DataFrame(
                        self.estimators_.get(d.name, self.fallback_).predict(
                            d[self.value_colnames_]
                        ),
                        index=d.index,
                    )
                )
                .values.squeeze()
            )
        except AttributeError:
            # Handle new groups
            culprits = set(X[self.group_colnames_].agg(func=tuple, axis=1)) - set(
                self.estimators_.keys()
            )

            if self.shrinkage is not None and self.use_global_model:
                # Remove the global group from the culprits because the user did not specify
                culprits = {culprit[1:] for culprit in culprits}

            raise ValueError(
                f"found a group(s) {culprits} in `.predict` that was not in `.fit`"
            )

    def __predict_shrinkage_groups(self, X):
        """Make predictions for all shrinkage groups"""
        # DataFrame with predictions for each hierarchy level, per row. Missing groups errors are thrown here.
        hierarchical_predictions = pd.concat(
            [
                pd.Series(self.__predict_group(X, level_columns))
                for level_columns in self.group_colnames_hierarchical_
            ],
            axis=1,
        )

        # This is a Series with values the tuples of hierarchical grouping
        prediction_groups = X[self.group_colnames_].agg(func=tuple, axis=1)

        # This is a Series of arrays
        shrinkage_factors = prediction_groups.map(self.shrinkage_factors_)

        # Convert the Series of arrays it to a DataFrame
        shrinkage_factors = pd.DataFrame.from_dict(shrinkage_factors.to_dict()).T

        return (hierarchical_predictions * shrinkage_factors).sum(axis=1)

    def predict(self, X):
        """
        Predict on new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        X = self.__prepare_input_data(X)
        self.__validate(X)

        check_is_fitted(
            self,
            [
                "estimators_",
                "groups_",
                "group_colnames_",
                "value_colnames_",
                "fallback_",
            ],
        )

        if self.shrinkage is None:
            return self.__predict_group(X, group_colnames=self.group_colnames_)
        else:
            return self.__predict_shrinkage_groups(X)


class OutlierRemover(TrainOnlyTransformerMixin, BaseEstimator):
    """
    Removes outliers (train-time only) using the supplied removal model.

    :param outlier_detector: must implement `fit` and `predict` methods
    :param refit: If True, fits the estimator during pipeline.fit().

    """

    def __init__(self, outlier_detector, refit=True):
        self.outlier_detector = outlier_detector
        self.refit = refit
        self.estimator_ = None

    def fit(self, X, y=None):
        self.estimator_ = clone(self.outlier_detector)
        if self.refit:
            super().fit(X, y)
            self.estimator_.fit(X, y)
        return self

    def transform_train(self, X):
        check_is_fitted(self, "estimator_")
        predictions = self.estimator_.predict(X)
        check_array(predictions, estimator=self.outlier_detector, ensure_2d=False)
        return X[predictions != -1]


class DecayEstimator(BaseEstimator):
    """
    Morphs an estimator suchs that the training weights can be
    adapted to ensure that points that are far away have less weight.
    Note that it is up to the user to sort the dataset appropriately.
    This meta estimator will only work for estimators that have a
    "sample_weights" argument in their `.fit()` method.

    The DecayEstimator will use exponential decay to weight the parameters.

    w_{t-1} = decay * w_{t}
    """

    def __init__(self, model, decay: float = 0.999, decay_func="exponential"):
        self.model = model
        self.decay = decay
        self.func = decay_func

    def _is_classifier(self):
        return any(
            ["ClassifierMixin" in p.__name__ for p in type(self.model).__bases__]
        )

    def fit(self, X, y):
        """
        Fit the data after adapting the same weight.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.weights_ = np.cumprod(np.ones(X.shape[0]) * self.decay)[::-1]
        self.estimator_ = clone(self.model)
        try:
            self.estimator_.fit(X, y, sample_weight=self.weights_)
        except TypeError as e:
            if "sample_weight" in str(e):
                raise TypeError(
                    f"Model {type(self.model).__name__}.fit() does not have 'sample_weight'"
                )
        if self._is_classifier():
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        if self._is_classifier():
            check_is_fitted(self, ["classes_"])
        check_is_fitted(self, ["weights_", "estimator_"])
        return self.estimator_.predict(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)


class Thresholder(BaseEstimator, ClassifierMixin):
    """
    Takes a two class estimator and moves the threshold. This way you might
    design the algorithm to only accept a certain class if the probability
    for it is larger than, say, 90% instead of 50%.
    """

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.estimator_ = clone(self.model)
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError(
                "The Thresholder meta model only works on classifcation models with .predict_proba."
            )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "The Thresholder meta model only works on models with two classes."
            )
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ["classes_", "estimator_"])
        predicate = self.estimator_.predict_proba(X)[:, 1] > self.threshold
        return np.where(predicate, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "estimator_"])
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)


class ConfusionBalancer(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    The ConfusionBalancer attempts to give it's child estimator a more balanced
    output by learning from the confusion matrix during training. The idea is that
    the confusion matrix calculates P(C_i | M_i) where C_i is the actual class and
    M_i is the class that the underlying model gives. We use these probabilities to
    attempt a more balanced prediction by averaging the correction from the confusion
    matrix with the original probabilities.

    .. math::
        p(\text{class_j}) = \alpha p(\text{model}_j) + (1-\alpha) p(\text{class_j} | \text{model}_j) p(\text{model}_j)

    :param model: a scikit learn compatible classification model that has predict_proba
    :param alpha: a hyperparameter between 0 and 1, determines how much to apply smoothing
    :param cfm_smooth: a smoothing parameter for the confusion matrices to ensure zeros don't exist
    """

    def __init__(self, estimator, alpha: float = 0.5, cfm_smooth=0):
        self.estimator = estimator
        self.alpha = alpha
        self.cfm_smooth = cfm_smooth

    def fit(self, X, y):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self.estimator, dtype=FLOAT_DTYPES)
        if not isinstance(self.estimator, ProbabilisticClassifier):
            raise ValueError(
                "The ConfusionBalancer meta model only works on classifcation models with .predict_proba."
            )
        self.estimator.fit(X, y)
        self.classes_ = unique_labels(y)
        cfm = confusion_matrix(y, self.estimator.predict(X)).T + self.cfm_smooth
        self.cfm_ = cfm / cfm.sum(axis=1).reshape(-1, 1)
        return self

    def predict_proba(self, X):
        """
        Predict new data, with probabilities

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples, n_classes) the predicted data
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        preds = self.estimator.predict_proba(X)
        return (1 - self.alpha) * preds + self.alpha * preds @ self.cfm_

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ["cfm_", "classes_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SubjectiveClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    Corrects predictions of the inner classifier by taking into account a (subjective) prior distribution of the
    classes.

    This can be useful when there is a difference in class distribution between the training data set and
    the real world. Using the confusion matrix of the inner classifier and the prior, the posterior probability for a
    class, given the prediction of the inner classifier, can be computed. The background for this posterior estimation
    is given `in this article <https://lucdemortier.github.io/articles/16/PerformanceMetrics>_`.

    Based on the `evidence` attribute, this meta estimator's predictions are based on simple weighing of the inner
    estimator's `predict_proba()` results, the posterior probabilities based on the confusion matrix, or a combination
    of the two approaches.

    :param estimator: An sklearn-compatible classifier estimator
    :param prior: A dict of class->frequency representing the prior (a.k.a. subjective real-world) class
    distribution. The class frequencies should sum to 1.
    :param evidence: A string indicating which evidence should be used to correct the inner estimator's predictions.
    Should be one of 'predict_proba', 'confusion_matrix', or 'both' (default). If `predict_proba`, the inner estimator's
    `predict_proba()` results are multiplied by the prior distribution. In case of `confusion_matrix`, the inner
    estimator's discrete predictions are converted to posterior probabilities using the prior and the inner estimator's
    confusion matrix (obtained from the train data used in `fit()`). In case of `both` (default), the the inner
    estimator's `predict_proba()` results are multiplied by the posterior probabilities.
    """

    def __init__(self, estimator, prior, evidence="both"):
        self.estimator = estimator
        self.prior = prior
        self.evidence = evidence

    def _likelihood(self, predicted_class, given_class, cfm):
        return cfm[given_class, predicted_class] / cfm[given_class, :].sum()

    def _evidence(self, predicted_class, cfm):
        return sum(
            [
                self._likelihood(predicted_class, given_class, cfm)
                * self.prior[self.classes_[given_class]]
                for given_class in range(cfm.shape[0])
            ]
        )

    def _posterior(self, y, y_hat, cfm):
        y_hat_evidence = self._evidence(y_hat, cfm)
        return (
            (
                self._likelihood(y_hat, y, cfm)
                * self.prior[self.classes_[y]]
                / y_hat_evidence
            )
            if y_hat_evidence > 0
            else self.prior[y]  # in case confusion matrix has all-zero column for y_hat
        )

    def fit(self, X, y):
        """
        Fits the inner estimator based on the data.

        Raises a `ValueError` if the `y` vector contains classes that are not specified in the prior, or if the prior is
        not a valid probability distribution (i.e. does not sum to 1).

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        if not isinstance(self.estimator, ClassifierMixin):
            raise ValueError(
                "Invalid inner estimator: the SubjectiveClassifier meta model only works on classification models"
            )

        if not np.isclose(sum(self.prior.values()), 1):
            raise ValueError(
                "Invalid prior: the prior probabilities of all classes should sum to 1"
            )

        valid_evidence_types = ["predict_proba", "confusion_matrix", "both"]
        if self.evidence not in valid_evidence_types:
            raise ValueError(
                f"Invalid evidence: the provided evidence should be one of {valid_evidence_types}"
            )

        X, y = check_X_y(X, y, estimator=self.estimator, dtype=FLOAT_DTYPES)
        if set(y) - set(self.prior.keys()):
            raise ValueError(
                f"Training data is inconsistent with prior: no prior defined for classes "
                f"{set(y) - set(self.prior.keys())}"
            )
        self.estimator.fit(X, y)
        cfm = confusion_matrix(y, self.estimator.predict(X))
        self.posterior_matrix_ = np.array(
            [
                [self._posterior(y, y_hat, cfm) for y_hat in range(cfm.shape[0])]
                for y in range(cfm.shape[0])
            ]
        )
        return self

    @staticmethod
    def _weighted_proba(weights, y_hat_probas):
        return normalize(weights * y_hat_probas, norm="l1")

    @staticmethod
    def _to_discrete(y_hat_probas):
        y_hat_discrete = np.zeros(y_hat_probas.shape)
        y_hat_discrete[
            np.arange(y_hat_probas.shape[0]), y_hat_probas.argmax(axis=1)
        ] = 1
        return y_hat_discrete

    def predict_proba(self, X):
        """
        Returns probability distribution of the class, based on the provided data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples, n_classes) the predicted data
        """
        check_is_fitted(self, ["posterior_matrix_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        y_hats = self.estimator.predict_proba(X)  # these are ignorant of the prior

        if self.evidence == "predict_proba":
            prior_weights = np.array([self.prior[klass] for klass in self.classes_])
            return self._weighted_proba(prior_weights, y_hats)
        else:
            posterior_probas = self._to_discrete(y_hats) @ self.posterior_matrix_.T
            return (
                self._weighted_proba(posterior_probas, y_hats)
                if self.evidence == "both"
                else posterior_probas
            )

    def predict(self, X):
        """
        Returns predicted class, based on the provided data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples, n_classes) the predicted data
        """
        check_is_fitted(self, ["posterior_matrix_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def classes_(self):
        return self.estimator.classes_
