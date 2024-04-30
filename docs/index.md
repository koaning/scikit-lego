# scikit-lego

![logo](_static/logo.png)

We love scikit learn but very often we find ourselves writing custom transformers, metrics and models.
The goal of this project is to attempt to consolidate these into a package that offers code quality/testing.
This project is a collaboration between multiple companies in the Netherlands.
Note that we're not formally affiliated with the scikit-learn project at all.

## Disclaimer

LEGO® is a trademark of the LEGO Group of companies which does not sponsor, authorize or endorse this project.
Also note this package, albeit designing to be used on top of scikit-learn, is not associated with that project in any formal manner.

The goal of the package is to allow you to joyfully build with new building blocks that are scikit-learn compatible.

## Installation

Install `scikit-lego` via pip with

```bash
pip install scikit-lego
```

For more installation options and details, check the [installation section][installation-section].

## Usage

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklego.transformers import RandomAdder

X, y = ...

mod = Pipeline([
    ("scale", StandardScaler()),
    ("random_noise", RandomAdder()),
    ("model", LogisticRegression(solver='lbfgs'))
])

_ = mod.fit(X, y)
...
```

To see more examples, please refer to the [user guide section][user-guide].

[installation-section]: installation
[user-guide]: user-guide/datasets
