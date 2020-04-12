Contribution
============

.. image:: _static/contribute.png
   :align: center

This project started becauser we saw people rewrite the same
transformers and estimators at clients over and over again. Our
goal is to have a place where more experimental building blocks
for scikit learn pipelines might exist. This means we're usually
open to ideas to add here but there are a few things to keep in mind.

Before You Make a New Feature
-----------------------------

1. Discuss the feature and implementation you want to add on Github_ before you write a PR for it.
2. Features need a somewhat general usecase. If the usecase is very niche it will be hard for us to consider maintaining it.
3. If you're going to add a feature consider if you could help out in the maintenance of it.

When Writing a New Feature
--------------------------

When writing a new feauture there's some more details with regard to
how scikit learn likes to have it's parts implemented. We will display the a
sample implementation of the `ColumnSelector` below. Please review all comments marked as Important.

.. code-block:: python

    from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
    from sklearn.utils import check_array, check_X_y
    from sklearn.utils.validation import FLOAT_DTYPES, check_random_state, check_is_fitted

    class PandasTypeSelector(BaseEstimator, TransformerMixin):
        TYPES = {'number', 'category', 'float', 'int', 'object', 'datetime', 'timedelta'}
        """
        Select columns in a pandas dataframe based on their dtype

        :param include: types to be included in the dataframe
        :param exclude: types to be exluded in the dataframe
        """
        def __init__(self, include=None, exclude=None):
            # Important: You can't use `*args` or `**kwargs` in the `__init__` method. `scikit-learn` uses the
            # methods' call signature to figure out what hyperparameters for the estimator are.

            # Important: keep the same name for the function argument and the attribute stored on self.
            # If you don't, the `get_params` method will try to fetch the attribute with the name it has in the
            # function signature, but as that one doesn't exist, it will return `None`. This will silently break
            # copying
            self.include = include
            self.exclude = exclude

        def fit(self, X, y=None):
            """
            Saves the column names for check during transform

            :param X: pandas dataframe to select dtypes out of
            :param y: not used in this class
            """
            self._check_X_for_type(X)
            # Important: Normal software engineering practices would have you put these kinds of parameter
            # checks or basic casts inside the `__init__` method. `scikit-learn` will break when you do this,
            # as the method it uses for cloning estimators (e.g. while doing gridsearch) involves setting parameters
            # directly, after the class has been constructed.
            if len(set(self.include) - self.TYPES) > 0:
                raise ValueError(f'Unrecognized type in `include`.'
                                  'Expected {self.TYPES}, got {set(self.include) - self.TYPES}')
            if len(set(self.exclude) - self.TYPES) > 0:
                raise ValueError(f'Unrecognized type in `exclude`.'
                                  'Expected {self.TYPES}, got {set(self.exclude) - self.TYPES}')


            # Important: variables that are 'learned' during the fitting process should always have a trailing underscore
            # Please don't initialize these features inside the `__init__`, but initialize them in the `fit` method
            self.type_columns_ = list(X.select_dtypes(include=self.include, exclude=self.exclude))
            self.X_dtypes_ = X.dtypes
            if len(self.type_columns_) == 0:
                raise ValueError(f'Provided type(s) results in empty dateframe')

            # Important: Always return self from the `fit` method
            return self


        # Important: `y=None` exists solely for compatibility reasons. It will be removed sometime in the future
        def transform(self, X, y=None):
            """
            Transforms pandas dataframe by (de)selecting columns based on their dtype

            :param X: pandas dataframe to select dtypes for
            """
            # Important: Check whether the variables you expected to learn during fit are indeed present
            check_is_fitted(self, ['type_columns_', 'X_dtypes_'])
            if (self.X_dtypes_ != X.dtypes).any():
                raise ValueError(f'Column dtypes were not equal during fit and transform. Fit types: \n'
                                 f'{self.X_dtypes_}\n'
                                 f'transform: \n'
                                 f'{X.dtypes}')

            self._check_X_for_type(X)
            transformed_df = X.select_dtypes(include=self.include, exclude=self.exclude)

            if set(list(transformed_df)) != set(self.type_columns_):
                raise ValueError(f'Columns were not equal during fit and transform')

            return transformed_df

        @staticmethod
        def _check_X_for_type(X):
            """Checks if input of the Selector is of the required dtype"""
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Provided variable X is not of type pandas.DataFrame")

There's a few good practices we observe here that we'd appreciate
seeing in pull requests. We want to re-use features from sklearn as much as possible.
In particular, for this example:

1. We inherit from the mixins found in sklearn.
2. We use the validation utils from sklearn in our object to confirm if the model is fitted, if the array going into the model is of the correct type and if the random state is appropriate.

Feel free to look at example implementations before writing your own from scratch.

Unit Tests
----------

We write unit tests on these objects to make sure that they will work in a Pipeline_. This must
be guaranteed. To facilitate this we have some "standard" tests that will check things like "do
we change the shape of the input"? If your transformer belongs here: feel free to add it.

.. _Pipeline: https://scikit-learn.org/stable/modules/compose.html
.. _Github: https://github.com/koaning/scikit-lego/issues
