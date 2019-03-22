from sklearn import base
from sklearn.base import clone


class GroupEstimator(base.BaseEstimator):
    """
        Parameters
        ----------
          model : sklearn.base.BaseEstimator
              The model to run in the data
          dataset : pandas.DataFrame
              The Pandas dataset
          col_name_sep : List of string
            The name of the column to group by the data

         Examples
        --------
            >>> from sklego import GroupEstimator
            >>> from sklearn import DummyClassifier
            >>> GBE = GroupEstimator(DummyClassifier, dataset, ["chicken_weight"])
            >>> GBE.fit(X,y).predictt(x,y)
                [[result_df1, result_df2, result_df3]]
    """

    def __init__(self, model, col_name_sep=None):
        self.base_model = model
        self.col_name_sep = col_name_sep

    def separate_dataset(self, X, y, col_name_sep):
        """
        Separate dataset based on the value of coloumns

        Parameters
        ---------
        X  : pandas.DataFrame
            The dataset
        y  : pandas.DataFrame
            The predicted class
        col_name_sep: list of String
            The column name to separate the dataset

        Return
        ---------
        X_groups : A list of pandas.dataFrame of group of X data
        y_groups : A list of pandas.dataFrame of group of y data
        """
        X_groups = [data for _, data in X.groupby(col_name_sep)]
        y_groups = [y.iloc[data.index] for data in X_groups]
        return X_groups, y_groups

    def fit(self, X, y):
        """
        Fit the base model into all the groupped data

        Parameters
        ---------
        X  : pandas.DataFrame
            The dataset
        y  : pandas.DataFrame
            The predicted class
        """
        self.X_groups, self.y_groups = self.separate_dataset(X, y, self.col_name_sep)
        self.models = [
            clone(self.base_model).fit(x_subgroup, y_subgroup)
            for x_subgroup, y_subgroup in zip(self.X_groups, self.y_groups)
        ]
        return self

    def predict(self, X):
        """
            Predict using all the group of models

             Parameters
            ---------
            X  : pandas.DataFrame
                The dataset
        """
        return [model.predict(X) for model in self.models]
