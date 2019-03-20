from sklearn.model_selection._split import _BaseKFold
from sklearn.cluster import k_means
import numpy as np
import pandas as pd
import warnings


class KMeansFold(_BaseKFold):
    """
    K-Means cross validator

    :param n_splits: Number of splits (clusters) to fold on
    :param kmeans_kwargs: Extra kwargs for k_means apart from n_clusters
    """

    def __init__(self, n_splits='default', k_means_kwargs=None):
        if n_splits == 'default':
            warnings.warn(f'n_splits not provided, will be set to default ({n_splits})')
            n_splits = 3

        if not k_means_kwargs:
            k_means_kwargs = {}

        super(KMeansFold, self).__init__(n_splits=n_splits,
                                         shuffle=False,
                                         random_state=None)

        self.kmeans_kwargs = k_means_kwargs

    def _iter_test_indices(self, X, y=None, groups=None):
        """
        Generator to iterate over the indices

        :param X: Data to perform folds on
        :param y: Not used but required for _BaseKfold
        :param groups: Not used but required for _BaseKfold
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        clusters = k_means(X, n_clusters=self.n_splits, **self.kmeans_kwargs)[1]

        for label in np.unique(clusters):
            yield np.where(clusters == label)[0]
