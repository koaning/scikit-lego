![](https://travis-ci.com/koaning/scikit-lego.svg?branch=master)

# scikit-lego

We love scikit learn but very often we find ourselves writing
custom transformers, metrics and models. The goal of this project
is to attempt to consolidate these into a package that offers 
code quality/testing. This project is a collaboration between
multiple companies in the Netherlands. 

## Installation 

Install `scikit-lego` via pip with 

```bash
pip install scikit-lego
```

Alternatively you can fork/clone and run: 

```bash
$ pip install --editable .
```

## Usage 

```python
from sklego.transformers import RandomAdder

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

...

mod = Pipeline([
    ("scale", StandardScaler()),
    ("pca", RandomAdder()),
    ("model", LogisticRegression(solver='lbfgs'))
])

...
```

## New Features 

We want to be rather open here in what we accept but we do demand three 
things before they become added to the project:

1. any new feature contributes towards a demonstratable real-world usecase
2. any new feature passes standard unit tests (we have a few for transformers and predictors)
3. the feature has been discussed in the issue list beforehand 