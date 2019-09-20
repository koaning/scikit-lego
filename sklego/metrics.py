import numpy as np
import warnings


def correlation_score(column):
    """
    The correlation score can score how well the estimator predictions correlate with a given column.
    This is especially usefull to use in situations where "fairness" is a theme.

    `correlation_score` takes a column on which to calculate the correlation and returns a metric function

    Usage:
    `correlation_score('gender')(clf, X, y)`


    :param column: Name of the column (when X is a dataframe) or the index of the column (when X is a numpy array).
    :return:
        A function which calculates the negative correlation between estimator.predict(X) and X[colum]
        (in gridsearch, larger is better and we want to typically punish correlation).
    """
    def correlation_metric(estimator, X, y_true=None):
        """Remember: X is the thing going *in* to your pipeline."""
        sensitive_col = X[:, column] if isinstance(X, np.ndarray) else X[column]
        return -np.abs(np.corrcoef(estimator.predict(X), sensitive_col)[1, 0])
    return correlation_metric


def p_percent_score(column, positive_target=1):
    r"""
    The p_percent score calculates the ratio between the probability of a positive outcome
    given the sensitive attribute (column) being true and the same probability given the
    sensitive attribute being false.

    .. math::
        \min \left(\frac{P(\hat{y}=1 | z=1)}{P(\hat{y}=1 | z=0)}, \frac{P(\hat{y}=1 | z=0)}{P(\hat{y}=1 | z=1)}\right)

    This is especially usefull to use in situations where "fairness" is a theme.

    Usage:
    `p_percent_score('gender')(clf, X, y)`

    source:
    - M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification

    :param column: Name of the column (when X is a dataframe) or the index of the column (when X is a numpy array).
    :param positive_target: The name of the class which is associated with a positive outcome
    :return: a function that calculates the p percent score for z = column
    """
    def impl(estimator, X, y_true=None):
        """Remember: X is the thing going *in* to your pipeline."""
        sensitive_col = X[:, column] if isinstance(X, np.ndarray) else X[column]

        if not np.all((sensitive_col == 0) | (sensitive_col == 1)):
            raise ValueError(f'neg_p_percent only supports binary indicator columns for `column`. '
                             f'Found values {np.unique(sensitive_col)}')

        y_hat = estimator.predict(X)
        y_given_z1 = y_hat[sensitive_col == 1]
        y_given_z0 = y_hat[sensitive_col == 0]
        p_y1_z1 = np.mean(y_given_z1 == positive_target)
        p_y1_z0 = np.mean(y_given_z0 == positive_target)

        # If we never predict a positive target for one of the subgroups, the model is by definition not
        # fair so we return 0
        if p_y1_z1 == 0:
            warnings.warn(f"No samples with y_hat == {positive_target} for {column} == 1, returning 0", RuntimeWarning)
            return 0

        if p_y1_z0 == 0:
            warnings.warn(f"No samples with y_hat == {positive_target} for {column} == 0, returning 0", RuntimeWarning)
            return 0

        p_percent = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
        return p_percent if not np.isnan(p_percent) else 1
    return impl
