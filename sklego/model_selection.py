from sklearn.model_selection._split import _BaseKFold
import numpy as np
import pandas as pd


class KlusterFoldValidation(_BaseKFold):
    """
    K-Means cross validator

    :param n_splits: Number of splits (clusters) to fold on
    :param cluster_method: Clustering method with fit_predict attribute
    """

    def __init__(self, random_state, cluster_method=None):
        if not hasattr(cluster_method, 'fit_predict'):
            raise TypeError('Provide a cluster method which supports fit_predict operation')

        super(KlusterFoldValidation, self).__init__(n_splits=3,
                                                    shuffle=False,
                                                    random_state=random_state)

        self.cluster_method = cluster_method

    def _iter_test_indices(self, X, y=None, groups=None):
        """
        Generator to iterate over the indices

        :param X: Data to perform folds on
        :param y: Not used but required for _BaseKfold
        :param groups: Not used but required for _BaseKfold
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        clusters = self.cluster_method.fit_predict(X)

        self.n_splits = len(np.unique(clusters))

        if self.n_splits == 0:
            raise ValueError(f'Clustering method resulted in {self.n_splits} cluster, too few for fold validation')

        for label in np.unique(clusters):
            yield np.where(clusters == label)[0]
