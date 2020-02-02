import numpy as np
import warnings

from typing import Callable


def correlation_score(column):
    """
    The correlation score can score how well the estimator predictions correlate with a given column.
    This is especially useful to use in situations where "fairness" is a theme.

    `correlation_score` takes a column on which to calculate the correlation and returns a metric function

    Usage:
    `correlation_score('gender')(clf, X, y)`


    :param column: Name of the column (when X is a dataframe) or the index of the column (when X is a numpy array).
    :return:
        A function which calculates the negative correlation between estimator.predict(X) and X[column]
        (in gridsearch, larger is better and we want to typically punish correlation).
    """

    def correlation_metric(estimator, X, y_true=None):
        """Remember: X is the thing going *in* to your pipeline."""
        sensitive_col = X[:, column] if isinstance(X, np.ndarray) else X[column]
        return -np.abs(np.corrcoef(estimator.predict(X), sensitive_col)[1, 0])

    return correlation_metric


def p_percent_score(sensitive_column, positive_target=1):
    r"""
    The p_percent score calculates the ratio between the probability of a positive outcome
    given the sensitive attribute (column) being true and the same probability given the
    sensitive attribute being false.

    .. math::
        \min \left(\frac{P(\hat{y}=1 | z=1)}{P(\hat{y}=1 | z=0)}, \frac{P(\hat{y}=1 | z=0)}{P(\hat{y}=1 | z=1)}\right)

    This is especially useful to use in situations where "fairness" is a theme.

    Usage:
    `p_percent_score('gender')(clf, X, y)`

    source:
    - M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification

    :param sensitive_column:
        Name of the column containing the binary sensitive attribute (when X is a dataframe)
        or the index of the column (when X is a numpy array).
    :param positive_target: The name of the class which is associated with a positive outcome
    :return: a function (clf, X, y_true) -> float that calculates the p percent score for z = column
    """

    def impl(estimator, X, y_true=None):
        """Remember: X is the thing going *in* to your pipeline."""
        sensitive_col = (
            X[:, sensitive_column] if isinstance(X, np.ndarray) else X[sensitive_column]
        )

        if not np.all((sensitive_col == 0) | (sensitive_col == 1)):
            raise ValueError(
                f"p_percent_score only supports binary indicator columns for `column`. "
                f"Found values {np.unique(sensitive_col)}"
            )

        y_hat = estimator.predict(X)
        y_given_z1 = y_hat[sensitive_col == 1]
        y_given_z0 = y_hat[sensitive_col == 0]
        p_y1_z1 = np.mean(y_given_z1 == positive_target)
        p_y1_z0 = np.mean(y_given_z0 == positive_target)

        # If we never predict a positive target for one of the subgroups, the model is by definition not
        # fair so we return 0
        if p_y1_z1 == 0:
            warnings.warn(
                f"No samples with y_hat == {positive_target} for {sensitive_column} == 1, returning 0",
                RuntimeWarning,
            )
            return 0

        if p_y1_z0 == 0:
            warnings.warn(
                f"No samples with y_hat == {positive_target} for {sensitive_column} == 0, returning 0",
                RuntimeWarning,
            )
            return 0

        p_percent = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
        return p_percent if not np.isnan(p_percent) else 1

    return impl


def equal_opportunity_score(sensitive_column, positive_target=1):
    r"""
    The equality opportunity score calculates the ratio between the probability of a **true positive** outcome
    given the sensitive attribute (column) being true and the same probability given the
    sensitive attribute being false.

    .. math::
        \min \left(\frac{P(\hat{y}=1 | z=1, y=1)}{P(\hat{y}=1 | z=0, y=1)},
        \frac{P(\hat{y}=1 | z=0, y=1)}{P(\hat{y}=1 | z=1, y=1)}\right)

    This is especially useful to use in situations where "fairness" is a theme.

    Usage:
    `equal_opportunity_score('gender')(clf, X, y)`

    Source:
    - M. Hardt, E. Price and N. Srebro (2016), Equality of Opportunity in Supervised Learning

    :param sensitive_column:
        Name of the column containing the binary sensitive attribute (when X is a dataframe)
        or the index of the column (when X is a numpy array).
    :param positive_target: The name of the class which is associated with a positive outcome
    :return: a function (clf, X, y_true) -> float that calculates the equal opportunity score for z = column
    """

    def impl(estimator, X, y_true):
        """Remember: X is the thing going *in* to your pipeline."""
        sensitive_col = (
            X[:, sensitive_column] if isinstance(X, np.ndarray) else X[sensitive_column]
        )

        if not np.all((sensitive_col == 0) | (sensitive_col == 1)):
            raise ValueError(
                f"equal_opportunity_score only supports binary indicator columns for `column`. "
                f"Found values {np.unique(sensitive_col)}"
            )

        y_hat = estimator.predict(X)
        y_given_z1_y1 = y_hat[(sensitive_col == 1) & (y_true == positive_target)]
        y_given_z0_y1 = y_hat[(sensitive_col == 0) & (y_true == positive_target)]

        # If we never predict a positive target for one of the subgroups, the model is by definition not
        # fair so we return 0
        if len(y_given_z1_y1) == 0:
            warnings.warn(
                f"No samples with y_hat == {positive_target} for {sensitive_column} == 1, returning 0",
                RuntimeWarning,
            )
            return 0

        if len(y_given_z0_y1) == 0:
            warnings.warn(
                f"No samples with y_hat == {positive_target} for {sensitive_column} == 0, returning 0",
                RuntimeWarning,
            )
            return 0

        p_y1_z1 = np.mean(y_given_z1_y1 == positive_target)
        p_y1_z0 = np.mean(y_given_z0_y1 == positive_target)
        score = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
        return score if not np.isnan(score) else 1

    return impl


def subset_score(subset_picker: Callable, score: Callable, **kwargs):
    r"""
    Returns a method that applies the passed score only to a specific subset. The subset picker
    is a method that is passed the corresponding X and y_true and returns a one-dimensional
    boolean vector where every element corresponds to a row in the data. Only the elements
    with a True value are taken into account for the passed score, representing a filter.

    This allows users to have an easy approach to measuring metrics over different slices of
    the population which can give insights into the model performance, either specifically for
    fairness or in general.

    Usage:
    `subset_score(lambda X, y_true: X['column'] == 'A', accuracy_score)(clf, X, y)`

    :param subset_picker: Method that returns a boolean mask that is used for slicing the samples
    :param score: The score that needs to be applied to the subset
    :param kwargs: Additional keyword arguments to pass to score
    :return: a function that calculates the passed score for the subset
    """

    def sliced_metric(estimator, X, y_true=None):
        mask = subset_picker(X, y_true)
        if isinstance(mask, np.ndarray):
            if len(mask.shape) > 1:
                raise ValueError(
                    "`subset_picker` should return 1-dimensional numpy array or Pandas"
                    + " series, returned {} instead".format(len(mask.shape))
                )
        if np.sum(mask) == 0:
            warnings.warn(f"No samples in subset, returning NaN", RuntimeWarning)
            return np.nan
        X = X[mask]
        y_pred = estimator.predict(X)
        return score(y_true[mask], y_pred, **kwargs)

    return sliced_metric
