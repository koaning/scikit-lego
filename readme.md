![](https://travis-ci.com/koaning/scikit-lego.svg?branch=master) [![Build status](https://ci.appveyor.com/api/projects/status/66r9jjs844v8c5qh?svg=true)](https://ci.appveyor.com/project/koaning/scikit-lego) [![Documentation Status](https://readthedocs.org/projects/scikit-lego/badge/?version=latest)](https://scikit-lego.readthedocs.io/en/latest/?badge=latest)



# scikit-lego

![](images/logo.png)

We love scikit learn but very often we find ourselves writing
custom transformers, metrics and models. The goal of this project
is to attempt to consolidate these into a package that offers 
code quality/testing. This project is a collaboration between
multiple companies in the Netherlands. Note that we're not formally 
affiliated with the scikit-learn project at all. 

## Installation 

Install `scikit-lego` via pip with 

```bash
pip install scikit-lego
```

Alternatively, to edit and contribute you can fork/clone and run: 

```bash
pip install -e ".[dev]"
python setup.py develop
```

## Documentation 

The documentation can be found [here](scikit-lego.readthedocs.io). 

## Usage 

```python
from sklego.preprocessing import RandomAdder
from sklego.mixture import GMMClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

...

mod = Pipeline([
    ("scale", StandardScaler()),
    ("random_noise", RandomAdder()),
    ("model", GMMClassifier(solver='lbfgs'))
])

...
```

## Features 

Here's a list of features that this library currently offers: 

- `sklego.preprocessing.PatsyTransformer` applies a [patsy](https://patsy.readthedocs.io/en/latest/formulas.html) formula
- `sklego.preprocessing.RandomAdder` adds randomness in training
- `sklego.meta.EstimatorTransformer` adds a model output as a feature
- `sklego.dummy.RandomRegressor` benchmark that predicts random values
- `sklego.mixture.GMMClassifier` classifies by training a GMM per class
- `sklego.mixture.GMMOutlierDetector` detects outliers based on a trained GMM
- `sklego.pandas_utils.log_step` a simple logger-decorator for pandas pipeline steps
- `sklego.pandas_utils.add_lags` adds lag values of certain columns in pandas 
- `sklego.selectors.PandasTypeSelector` selects columns based on pandas type
- `sklego.selectors.ColumnSelector` selects columns based on column name
- `sklego.pipeline.DebugPipeline` adds debug information to make debugging easier
- `sklego.meta.GroupedEstimator` can split the data into runs and run a model on each
- `sklego.datasets.load_chicken` loads in the joyful chickweight dataset 

## New Features 

We want to be rather open here in what we accept but we do demand three 
things before they become added to the project:

1. any new feature contributes towards a demonstratable real-world usecase
2. any new feature passes standard unit tests (we have a few for transformers and predictors)
3. the feature has been discussed in the issue list beforehand 
