import numpy as np


def correlation_score(column):
    """
    The correlation score can score how well the estimator predictions
    correlate with a given column. This is especially usefull to use
    in situations where "fairness" is a theme.
    :param column: Name of the column (when X is a dataframe)
    or the index of the column (when X is a numpy array).
    :return: Negative correlation between estimator.predict(X) and X[colum]
    (in gridsearch, larger is better and we want to typically punish correlation).
    """
    def fairness_metric(estimator, X, y_true=None):
        """Remember: X is the thing going *in* to your pipeline."""
        if isinstance(X, np.ndarray):
            return -np.abs(np.corrcoef(estimator.predict(X), X[:, column])[1, 0])
        return -np.abs(np.corrcoef(estimator.predict(X), X[column])[1, 0])
    return fairness_metric
