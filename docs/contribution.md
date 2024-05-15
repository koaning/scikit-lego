# Contribution

<p align="center">
  <img src="../_static/contribution/contribute.png" />
</p>

This project started because we saw people rewrite the same transformers and estimators at clients over and over again.
Our goal is to have a place where more experimental building blocks for scikit learn pipelines might exist.
This means we're usually open to ideas to add here but there are a few things to keep in mind.

## Before You Make a New Feature

1. Discuss the feature and implementation you want to add on [Github][gh-issues]
    before you write a PR for it.
2. Features need a somewhat general usecase.
    If the usecase is very niche it will be hard for us to consider maintaining it.
3. If you're going to add a feature consider if you could help out in the maintenance of it.

## When Writing a New Feature

When writing a new feature there's some more
[details with regard to how scikit learn likes to have its parts implemented][scikit-develop].
We will display a sample implementation of the `ColumnSelector` below. Please review all comments marked as Important.

```py hl_lines="19-22 24-28 46-51 65-69 77-78 83-85" linenums="1"
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES, check_random_state, check_is_fitted

class TypeSelector(BaseEstimator, TransformerMixin):
    TYPES = {'number', 'category', 'float', 'int', 'object', 'datetime', 'timedelta'}
    """
    Select columns in a pandas dataframe based on their dtype

    Parameters
    ----------
    include : scalar or list-like
        Column type(s) to be selected
    exclude : scalar or list-like
        Column type(s) to be excluded from selection
    """
    def __init__(self, include=None, exclude=None):
        """
        Important:
            You can't use `*args` or `**kwargs` in the `__init__` method.
            scikit-learn uses the methods' call signature to figure out what
            hyperparameters for the estimator are.

        Important:
            Keep the same name for the function argument and the attribute stored on self.
            If you don't, the `get_params` method will try to fetch the attribute with
            the name it has in the function signature, but as that one doesn't exist,
            it will return `None`. This will silently break copying.
        """
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        """
        Saves the column names for check during transform

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.
        y : pd.Series, default=None
            Ignored, present for compatibility.
        """
        self._check_X_for_type(X)
        """
        Important:
            Normal software engineering practices would have you put these kinds of
            parameter checks or basic casts inside the `__init__` method.
            `scikit-learn` will break when you do this, as the method it uses for cloning
            estimators (e.g. while doing gridsearch) involves setting parameters directly,
            after the class has been constructed.
        """
        if len(set(self.include) - self.TYPES) > 0:
            raise ValueError(
                f'Unrecognized type in `include`.'
                'Expected {self.TYPES}, got {set(self.include) - self.TYPES}'
            )
        if len(set(self.exclude) - self.TYPES) > 0:
            raise ValueError(
                f'Unrecognized type in `exclude`.'
                'Expected {self.TYPES}, got {set(self.exclude) - self.TYPES}'
            )

        """
        Important:
            variables that are 'learned' during the fitting process should always have a
            trailing underscore.
            Please don't initialize these features inside the `__init__`, but initialize
            them in the `fit` method.
        """
        self.type_columns_ = list(X.select_dtypes(include=self.include, exclude=self.exclude))
        self.X_dtypes_ = X.dtypes
        if len(self.type_columns_) == 0:
            raise ValueError(f'Provided type(s) results in empty dateframe')

        """
        Important:
            Always return self from the `fit` method
        """
        return self

    """
    Important:
        `y=None` exists solely for compatibility reasons. It will be removed sometime
        in the future.
    """
    def transform(self, X, y=None):
        """
        Transforms pandas dataframe by (de)selecting columns based on their dtype

        Parameters
        ----------
        X : pd.DataFrame
            The data to select dtype for.
        """
        # Important: Check whether the variables you expected to learn during fit are indeed present
        check_is_fitted(self, ['type_columns_', 'X_dtypes_'])
        if (self.X_dtypes_ != X.dtypes).any():
            raise ValueError(
                f'Column dtypes were not equal during fit and transform. Fit types: \n'
                f'{self.X_dtypes_}\n'
                'transform: \n'
                f'{X.dtypes}'
            )

        self._check_X_for_type(X)
        transformed_df = X.select_dtypes(include=self.include, exclude=self.exclude)

        if set(list(transformed_df)) != set(self.type_columns_):
            raise ValueError('Columns were not equal during fit and transform')

        return transformed_df

    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")
```

There's a few good practices we observe here that we'd appreciate seeing in pull requests.
We want to re-use features from sklearn as much as possible.

In particular, for this example:

1. We inherit from the mixins found in sklearn.
2. We use the validation utils from sklearn in our object to confirm if the model is fitted, if the array going into
    the model is of the correct type and if the random state is appropriate.

Feel free to look at example implementations before writing your own from scratch.

## Unit Tests

We write unit tests on these objects to make sure that they will work in a [scikit-learn Pipeline][pipe-api].

**This must be guaranteed**. To facilitate this we have some _standard_ tests that will check things like
_"do we change the shape of the input?"_.

If your transformer belongs here: feel free to add it.

## Documentation

The documentation is generated using [Material for MkDocs][mkdocs-material], its extensions and a few plugins.
In particular [`mkdocstrings-python`][mkdocstrings-python] is used for API rendering.

When a new feature is introduced, it should be documented, and typically there are a few files to add or edit:

- [x] A page in the `docs/api/` folder.
- [x] A user guide in the `docs/user-guide/` folder.
- [x] A python script in the `docs/_scripts/` folder to generate plots and code snippets (see [next section](#working-with-pymdown-snippets-extension))
- [x] Relevant static files, such as images, plots, tables and html's, should be saved in the `docs/_static/` folder.
- [x] Edit the `mkdocs.yaml` file to include the new pages in the navigation.

### Working with pymdown snippets extension

The majority of code and code generate plots in the documentation is generated using the scripts in the `docs/_scripts/` folder, and accessed via the [pymdown snippets][pymdown-snippets] extension.

The reason for this separation is that:

- Markdowns are significantly easier to maintain and review than notebooks.
- Embedding code directly into markdown is simple and convenient, however it does not generate outputs.
- Instead of having duplicated (and possibly out of sync) code in markdown for rendering and in notebooks/scripts to generate outputs, via [pymdown snippets][pymdown-snippets] extension we can bind the two together.
- To generate the plots and/or results of a given section it is enough to run the corresponding script from the `docs/_scripts/` folder.

    ```bash
    cd docs
    python _scripts/<filename>.py
    ```

!!! info

    To generate all the outputs and static files from scratch it is enough to run the following command from the root of the repository:
    ```bash
    cd docs
    make generate-all
    ```
    which will run all the scripts and save results in the `docs/_static` folder.

### Render locally

The first step to render the documentation locally is to install the required dependencies:

```bash
python -m pip install -e ."[docs]"
```

Then from the root of the project, there are two options:

=== "mkdocs directly"

    ```bash
    mkdocs serve
    ```

=== "via make"

    ```bash
    make docs
    ```

!!! info

    Using mkdocs directly will allow to add extra params to the command if needed.

Then the documentation page will be available at [localhost][localhost].

[gh-issues]: https://github.com/koaning/scikit-lego/issues
[scikit-develop]: https://scikit-learn.org/stable/developers/develop.html
[pipe-api]: https://scikit-learn.org/stable/modules/compose.html
[mkdocs-material]: https://squidfunk.github.io/mkdocs-material/
[mkdocstrings-python]: https://mkdocstrings.github.io/python/
[pymdown-snippets]: https://facelessuser.github.io/pymdown-extensions/extensions/snippets/
[localhost]: http://localhost:8000/
