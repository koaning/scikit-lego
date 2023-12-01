from pathlib import Path

import gif

# --8<-- [start:common-imports]
import matplotlib.pylab as plt
import seaborn as sns
sns.set_theme()
# --8<-- [end:common-imports]

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

######################################## Lowess ##########################################
##########################################################################################

# --8<-- [start:lowess]
import numpy as np
from sklego.linear_model import LowessRegression

n = 100
xs = np.linspace(0, np.pi, n)
ys = 1 + np.sin(xs) + np.cos(xs**2) + np.random.normal(0, 0.1, n)

mod = LowessRegression(sigma=0.1).fit(xs.reshape(-1, 1), ys)

xs_new = np.linspace(-1, np.pi + 1, n * 2)
preds = mod.predict(xs_new.reshape(-1, 1))
# --8<-- [end:lowess]

# --8<-- [start:plot-lowess]
plt.figure(figsize=(12, 4))
plt.scatter(xs, ys)
plt.plot(xs_new, preds, color="orange")
plt.title("Be careful with extrapolation here.")
# --8<-- [start:plot-lowess]

plt.savefig(_static_path / "lowess.png")
plt.clf()


def plot_grid_weights(sigmas, spans):
    n, m = len(sigmas), len(spans)
    _, axes = plt.subplots(n, m, figsize=(10, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        span = spans[i % m]
        sigma = sigmas[int(i / n) % m]
        mod = LowessRegression(sigma=sigma, span=span).fit(xs.reshape(-1, 1), ys)
        wts = mod._calc_wts([1.5])
        ax.plot(xs, wts, color="steelblue")
        ax.set_title(f"$\sigma$={sigma}, span={span}")
    return axes


fig = plot_grid_weights(sigmas=[1.0, 0.1], spans=[0.1, 0.9])

plt.savefig(_static_path / "grid-span-sigma-01.png")
plt.clf()


def plot_spansigma(sigmas, spans):
    n, m = len(sigmas), len(spans)
    _, axes = plt.subplots(n, m, figsize=(10, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        span = spans[i % m]
        sigma = sigmas[int(i / n) % m]
        mod = LowessRegression(sigma=sigma, span=span).fit(xs.reshape(-1, 1), ys)
        preds = mod.predict(xs_new.reshape(-1, 1))
        ax.scatter(xs, ys)
        ax.plot(xs_new, preds, color="orange")
        ax.set_title(f"$\sigma$={sigma}, span={span}")
    return axes


fig = plot_spansigma(sigmas=[1.0, 0.1], spans=[0.1, 0.9])
plt.savefig(_static_path / "grid-span-sigma-02.png")
plt.clf()


@gif.frame
def single_frame(i, sigma, with_pred=False):
    mod = LowessRegression(sigma=sigma).fit(xs.reshape(-1, 1), ys)
    preds = mod.predict(xs.reshape(-1, 1))
    plt.figure(figsize=(10, 3))
    wts = mod._calc_wts(xs[i])
    plt.scatter(xs, ys, color="gray")
    plt.plot(xs, wts, color="red", alpha=0.5)
    for j in range(len(xs)):
        plt.scatter([xs[j]], [ys[j]], alpha=wts[j], color="orange")
    if with_pred:
        plt.plot(xs[:i], preds[:i], color="red")
    plt.title(f"$\sigma$={sigma}")


for sigma, name, with_pred in zip((0.1, 0.1, 0.01), ("01", "01", "001"), (False, True, True)):
    frames = [single_frame(i, sigma, with_pred=with_pred) for i in range(100)]
    suffix = f"{'-' + name if with_pred else ''}"
    gif.save(frames, str(_static_path / f"lowess-rolling{suffix}.gif"), duration=100)


n = 100
xs_orig = xs_sparse = np.linspace(0, np.pi, n)

ys_sparse = 1 + np.sin(xs_sparse) + np.cos(xs_sparse**2) + np.random.normal(0, 0.1, n)
keep = (xs_sparse < 0.8) | (xs_sparse > 1.6)

xs_sparse, ys_sparse = xs_sparse[keep], ys_sparse[keep]

mod_small = LowessRegression(sigma=0.01).fit(xs.reshape(-1, 1), ys)
mod_big = LowessRegression(sigma=0.1).fit(xs.reshape(-1, 1), ys)

preds_small = mod_small.predict(xs_orig.reshape(-1, 1))
preds_big = mod_big.predict(xs_orig.reshape(-1, 1))


@gif.frame
def double_frame(i):
    plt.figure(figsize=(10, 3))
    wts_small = mod_small._calc_wts(xs_orig[i])
    wts_big = mod_big._calc_wts(xs_orig[i])

    plt.scatter(xs_sparse, ys_sparse, color="gray")

    plt.plot(xs_orig, wts_big, color="green", alpha=0.5)
    plt.plot(xs_orig, wts_small, color="red", alpha=0.5)

    plt.plot(xs_orig[:i], preds_big[:i], color="green", label="$\sigma$=0.1")
    plt.plot(xs_orig[:i], preds_small[:i], color="red", label="$\sigma$=0.01")

    plt.legend()


frames = [double_frame(i) for i in range(len(xs))]
gif.save(frames, str(_static_path / "lowess-two-predictions.gif"), duration=100)

################################# ProbWeightRegression ###################################
##########################################################################################

# --8<-- [start:prob-weight-data]
from sklearn.datasets import make_regression
import pandas as pd

X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
df = pd.DataFrame(X)
# --8<-- [end:prob-weight-data]

# --8<-- [start:prob-weight-regr]
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklego.meta import EstimatorTransformer
from sklego.linear_model import ProbWeightRegression
from sklego.preprocessing import ColumnSelector

pipe = Pipeline([
    ("models", FeatureUnion([
        ("path1", Pipeline([
            ("select1", ColumnSelector([0, 1, 2, 3, 4])),
            ("pca", PCA(n_components=3)),
            ("linear", EstimatorTransformer(LinearRegression()))
        ])),
        ("path2", Pipeline([
            ("select2", ColumnSelector([5,6,7,8,9])),
            ("pca", PCA(n_components=2)),
            ("linear", EstimatorTransformer(LinearRegression()))
        ]))
    ])),
    ("prob_weight", ProbWeightRegression())
])

grid = GridSearchCV(estimator=pipe, param_grid={}, cv=3).fit(df, y)
# --8<-- [end:prob-weight-regr]

# --8<-- [start:prob-weight-display]
from sklearn import set_config
set_config(display="diagram")
grid
# --8<-- [end:prob-weight-display]

from sklearn.utils import estimator_html_repr
with open(_static_path / "grid.html", "w") as f:
    f.write(estimator_html_repr(grid))

# --8<-- [start:prob-weight-coefs]
grid.best_estimator_[1].coefs_
# array([0.03102466, 0.96897535])
# --8<-- [end:prob-weight-coefs]

#################################### LADRegression #######################################
##########################################################################################

# --8<-- [start:lad-data]
import numpy as np

np.random.seed(0)
X = np.linspace(0, 1, 20)
y = 3 * X + 1 + 0.5 * np.random.randn(20)
X = X.reshape(-1, 1)

y[10] = 8
y[15] = 15

plt.figure(figsize=(16, 4))
plt.scatter(X, y)
# --8<-- [end:lad-data]

plt.savefig(_static_path / "lad-data.png")
plt.clf()

# --8<-- [start:lr-fit]
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([0, 1]).reshape(-1, 1)
plt.figure(figsize=(16, 4))
plt.scatter(X, y)
plt.plot(x, LinearRegression().fit(X, y).predict(x), "r");
# --8<-- [end:lr-fit]

plt.savefig(_static_path / "lr-fit.png")
plt.clf()

# --8<-- [start:lad-fit]
import numpy as np
from sklearn.linear_model import LinearRegression
from sklego.linear_model import LADRegression

x = np.array([0, 1]).reshape(-1, 1)
plt.figure(figsize=(16, 4))
plt.scatter(X, y)
plt.plot(x, LinearRegression().fit(X, y).predict(x), "r");
plt.plot(x, LADRegression().fit(X, y).predict(x), "g");
# --8<-- [end:lad-fit]

plt.savefig(_static_path / "lad-fit.png")
plt.clf()

################################# QuantileRegression #####################################
##########################################################################################

# --8<-- [start:quantile-fit]
import numpy as np
from sklego.linear_model import QuantileRegression

np.random.seed(123)
X = np.arange(100).reshape(-1, 1)
y = 2*X.ravel() + X.ravel()*np.random.standard_cauchy(100)

q_10 = QuantileRegression(quantile=0.1).fit(X, y)
q_90 = QuantileRegression(quantile=0.9).fit(X, y)
lad = QuantileRegression().fit(X, y)

plt.plot(X, y)
plt.plot(X, lad.predict(X))
plt.fill_between(X.ravel(), q_10.predict(X), q_90.predict(X), alpha=0.33, color="orange");
# --8<-- [end:quantile-fit]

plt.savefig(_static_path / "quantile-fit.png")
plt.clf()
