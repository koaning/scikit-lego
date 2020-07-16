from sklearn.base import BaseEstimator, TransformerMixin


class GroupedTransformer(BaseEstimator, TransformerMixin):
    """
    Construct a transformer per data group. Splits data by groups from single or multiple columns
    and transforms remaining columns using the transformers corresponding to the groups.

    :param transformer: the transformer to be applied per group
    :param groups: the column(s) of the matrix/dataframe to select as a grouping parameter set
    :param use_global_model: Whether or not to fall back to a general transformation in case a group
                             is not found during `.transform()`
    """

    def __init__(self, transformer, groups=0, use_global_model=True):
        self.transformer = transformer
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X
