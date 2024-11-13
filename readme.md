[![Downloads](https://static.pepy.tech/badge/scikit-lego/month)](https://www.pepy.tech/projects/scikit-lego)
[![Version](https://img.shields.io/pypi/v/scikit-lego)](https://pypi.org/project/scikit-lego/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scikit-lego.svg)](https://anaconda.org/conda-forge/scikit-lego)
![](https://img.shields.io/github/license/koaning/scikit-lego)
![](https://img.shields.io/pypi/pyversions/scikit-lego)
![](https://img.shields.io/github/contributors/koaning/scikit-lego)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/166836939.svg)](https://zenodo.org/badge/latestdoi/166836939)

# scikit-lego

<a href="https://koaning.github.io/scikit-lego/"><img src="images/logo.png" width="35%" height="35%" align="right" /></a>

We love scikit learn but very often we find ourselves writing
custom transformers, metrics and models. The goal of this project
is to attempt to consolidate these into a package that offers
code quality/testing. This project started as a collaboration between
multiple companies in the Netherlands but has since received contributions
from around the globe. It was initiated by [Matthijs Brouns](https://www.mbrouns.com/)
and [Vincent D. Warmerdam](https://koaning.io) as a tool to teach people how
to contribute to open source.

Note that we're not formally affiliated with the scikit-learn project at all,
but we aim to strictly adhere to their standards.

The same holds with lego. LEGO® is a trademark of the LEGO Group of companies which does not sponsor, authorize or endorse this project.

## Installation

Install `scikit-lego` via pip with

```bash
python -m pip install scikit-lego
```

Via [conda](https://conda.io/projects/conda/en/latest/) with

```bash
conda install -c conda-forge scikit-lego
```

Alternatively, to edit and contribute you can fork/clone and run:

```bash
python -m pip install -e ".[dev]"
python setup.py develop
```

## Documentation

The documentation can be found [here](https://koaning.github.io/scikit-lego/).

## Usage

We offer custom metrics, models and transformers. You can import them just like you would
in scikit-learn.

```python
# the scikit learn stuff we love
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# from scikit lego stuff we add
from sklego.preprocessing import RandomAdder
from sklego.mixture import GMMClassifier

...

mod = Pipeline([
    ("scale", StandardScaler()),
    ("random_noise", RandomAdder()),
    ("model", GMMClassifier())
])

...
```

## Features

Here's a list of features that this library currently offers:

- `sklego.datasets.load_abalone` loads in the abalone dataset
- `sklego.datasets.load_arrests` loads in a dataset with fairness concerns
- `sklego.datasets.load_chicken` loads in the joyful chickweight dataset
- `sklego.datasets.load_heroes` loads a heroes of the storm dataset
- `sklego.datasets.load_hearts` loads a dataset about hearts
- `sklego.datasets.load_penguins` loads a lovely dataset about penguins
- `sklego.datasets.fetch_creditcard` fetch a fraud dataset from openml
- `sklego.datasets.make_simpleseries` make a simulated timeseries
- `sklego.pandas_utils.add_lags` adds lag values in a pandas dataframe
- `sklego.pandas_utils.log_step` a useful decorator to log your pipeline steps
- `sklego.dummy.RandomRegressor` dummy benchmark that predicts random values
- `sklego.linear_model.DeadZoneRegressor` experimental feature that has a deadzone in the cost function
- `sklego.linear_model.DemographicParityClassifier` logistic classifier constrained on demographic parity
- `sklego.linear_model.EqualOpportunityClassifier` logistic classifier constrained on equal opportunity
- `sklego.linear_model.ProbWeightRegression` linear model that treats coefficients as probabilistic weights
- `sklego.linear_model.LowessRegression` locally weighted linear regression
- `sklego.linear_model.LADRegression` least absolute deviation regression
- `sklego.linear_model.QuantileRegression` linear quantile regression, generalizes LADRegression
- `sklego.linear_model.ImbalancedLinearRegression` punish over/under-estimation of a model directly
- `sklego.naive_bayes.GaussianMixtureNB` classifies by training a 1D GMM per column per class
- `sklego.naive_bayes.BayesianGaussianMixtureNB` classifies by training a bayesian 1D GMM per class
- `sklego.mixture.BayesianGMMClassifier` classifies by training a bayesian GMM per class
- `sklego.mixture.BayesianGMMOutlierDetector` detects outliers based on a trained bayesian GMM
- `sklego.mixture.GMMClassifier` classifies by training a GMM per class
- `sklego.mixture.GMMOutlierDetector` detects outliers based on a trained GMM
- `sklego.meta.ConfusionBalancer` experimental feature that allows you to balance the confusion matrix
- `sklego.meta.DecayEstimator` adds decay to the sample_weight that the model accepts
- `sklego.meta.EstimatorTransformer` adds a model output as a feature
- `sklego.meta.OutlierClassifier` turns outlier models into classifiers for gridsearch
- `sklego.meta.GroupedPredictor` can split the data into runs and run a model on each
- `sklego.meta.GroupedTransformer` can split the data into runs and run a transformer on each
- `sklego.meta.SubjectiveClassifier` experimental feature to add a prior to your classifier
- `sklego.meta.Thresholder` meta model that allows you to gridsearch over the threshold
- `sklego.meta.RegressionOutlierDetector` meta model that finds outliers by adding a threshold to regression
- `sklego.meta.ZeroInflatedRegressor` predicts zero or applies a regression based on a classifier
- `sklego.preprocessing.ColumnCapper` limits extreme values of the model features
- `sklego.preprocessing.ColumnDropper` drops a column from pandas
- `sklego.preprocessing.ColumnSelector` selects columns based on column name
- `sklego.preprocessing.InformationFilter` transformer that can de-correlate features
- `sklego.preprocessing.IdentityTransformer` returns the same data, allows for concatenating pipelines
- `sklego.preprocessing.OrthogonalTransformer` makes all features linearly independent
- `sklego.preprocessing.TypeSelector` selects columns based on type
- `sklego.preprocessing.RandomAdder` adds randomness in training
- `sklego.preprocessing.RepeatingBasisFunction` repeating feature engineering, useful for timeseries
- `sklego.preprocessing.DictMapper` assign numeric values on categorical columns
- `sklego.preprocessing.OutlierRemover` experimental method to remove outliers during training
- `sklego.preprocessing.MonotonicSplineTransformer` re-uses `SplineTransformer` in an attempt to make monotonic features
- `sklego.model_selection.GroupTimeSeriesSplit` timeseries Kfold for groups with different amount of observations per group
- `sklego.model_selection.KlusterFoldValidation` experimental feature that does K folds based on clustering
- `sklego.model_selection.TimeGapSplit` timeseries Kfold with a gap between train/test
- `sklego.pipeline.DebugPipeline` adds debug information to make debugging easier
- `sklego.pipeline.make_debug_pipeline` shorthand function to create a debugable pipeline
- `sklego.metrics.correlation_score` calculates correlation between model output and feature
- `sklego.metrics.equal_opportunity_score` calculates equal opportunity metric
- `sklego.metrics.p_percent_score` proxy for model fairness with regards to sensitive attribute
- `sklego.metrics.subset_score` calculate a score on a subset of your data (meant for fairness tracking)

## New Features

We want to be rather open here in what we accept but we do demand three
things before they become added to the project:

1. any new feature contributes towards a demonstrable real-world usecase
2. any new feature passes standard unit tests (we use the ones from scikit-learn)
3. the feature has been discussed in the issue list beforehand

We automate all of our testing and use pre-commit hooks to keep the code working.
