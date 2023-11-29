from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

######################################## EstimatorTransformer ###########################################
##########################################################################################

# --8<-- [start:estimator-transformer-1]
import numpy as np
import pandas as pd

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LinearRegression, Ridge

from sklego.meta import EstimatorTransformer
from sklego.preprocessing import ColumnSelector

np.random.seed(42)
n = 1000
X = np.random.uniform(0, 1, (n, 2))
y = X.sum(axis=1) + np.random.uniform(0, 1, (n,))
df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})

pipeline = Pipeline([
    ("grab_columns", ColumnSelector(["x1", "x2"])),
    ("ml_features", FeatureUnion([
        ("model_1",  EstimatorTransformer(LinearRegression())),
        ("model_2",  EstimatorTransformer(Ridge()))
    ]))
])

pipeline.fit(df, y).transform(df)
# --8<-- [end:estimator-transformer-1]
print(pipeline.fit(df, y).transform(df))


# --8<-- [start:estimator-transformer-2]
pipeline = Pipeline([
    ("grab_columns", ColumnSelector(["x1", "x2"])),
    ("ml_features", FeatureUnion([
        ("p1", Pipeline([
            ("grab1", ColumnSelector(["x1"])),
            ("mod1", EstimatorTransformer(LinearRegression()))
        ])),
        ("p2", Pipeline([
            ("grab2", ColumnSelector(["x2"])),
            ("mod2", EstimatorTransformer(LinearRegression()))
        ]))
    ]))
])

pipeline.fit(df, y).transform(df)
# --8<-- [end:estimator-transformer-2]
print(pipeline.fit(df, y).transform(df))

######################################## IdentityTransformer ###########################################
##########################################################################################

# --8<-- [start:identity-transformer]
import numpy as np

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA

from sklego.preprocessing import IdentityTransformer
np.random.seed(42)

n = 100
X = np.random.uniform(0, 1, (n, 4))

pipeline = Pipeline([
    ("split", FeatureUnion([
        ("orig", IdentityTransformer()),
        ("pca", PCA(2)),
    ]))
])

X_new = pipeline.fit_transform(X)
# --8<-- [end:identity-transformer]

print(np.round(X_new[:3], 4))
print(np.round(X[:3], 4))

######################################## ColumnCapper ###########################################
##########################################################################################

# --8<-- [start:column-capper]
import numpy as np
from sklego.preprocessing import ColumnCapper

np.random.seed(42)
X = np.random.uniform(0, 1, (100000, 2))

cc = ColumnCapper()
output = cc.fit(X).transform(X)
print(f"min capped at  5th quantile: {output.min(axis=0)}")
print(f"max capped at 95th quantile: {output.max(axis=0)}")

cc = ColumnCapper(quantile_range=(10, 90))
output = cc.fit(X).transform(X)
print(f"min capped at 10th quantile: {output.min(axis=0)}")
print(f"max capped at 90th quantile: {output.max(axis=0)}")

# min capped at  5th quantile: [0.05120598 0.0502972 ]
# max capped at 95th quantile: [0.94966328 0.94964339]
# min capped at 10th quantile: [0.10029693 0.09934085]
# max capped at 90th quantile: [0.90020412 0.89859006]
# --8<-- [end:column-capper]

# --8<-- [start:column-capper-inf]
arr = np.array([[0.0, np.inf],
                [-np.inf, 1.0]])
cc.transform(arr)
# --8<-- [end:column-capper-inf]


######################################## Patsy ###########################################
##########################################################################################

# --8<-- [start:patsy-1]
import pandas as pd
from sklego.preprocessing import PatsyTransformer

df = pd.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": ["yes", "yes", "no", "maybe", "yes"],
    "y": [2, 2, 4, 4, 6]
})
X, y = df[["a", "b"]], df[["y"]].to_numpy()

pt = PatsyTransformer("a + np.log(a) + b")
pt.fit(X, y).transform(X)
# --8<-- [end:patsy-1]

# --8<-- [start:patsy-2]
pt = PatsyTransformer("a + np.log(a) + b - 1")
pt.fit(X, y).transform(X)
# --8<-- [end:patsy-2]


######################################## RBF ###########################################
##########################################################################################

# --8<-- [start:rbf-data]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# generate features
X = pd.DataFrame({
    "day": np.arange(4*365),
    "day_of_year": (1 + np.arange(4*365)) % 365
})

# generate target
signal1 = 4 + 3*np.sin(X["day"]/365*2*np.pi)
signal2 = 4 * np.sin(X["day"]/365*4*np.pi+365/2)
noise = np.random.normal(0, 0.9, len(X["day"]))
y = signal1 + signal2 + noise

# plot
fig = plt.figure(figsize=(17,3))
ax = fig.add_subplot(111)
ax.plot(X["day"], y);
# --8<-- [end:rbf-data]

plt.savefig(_static_path / "rbf-data.png")
plt.clf()

# --8<-- [start:rbf-transform]
from sklego.preprocessing import RepeatingBasisFunction

N_PERIODS = 5
rbf = RepeatingBasisFunction(
    n_periods=N_PERIODS,
    remainder="passthrough",
    column="day_of_year",
    input_range=(1,365)
)

_ = rbf.fit(X)
Xt = rbf.transform(X)
# --8<-- [end:rbf-transform]


# --8<-- [start:rbf-plot]
fig, axes = plt.subplots(nrows=Xt.shape[1], figsize=(17,12))
for i in range(Xt.shape[1]):
    axes[i].plot(X["day"], Xt[:,i])
# --8<-- [end:rbf-plot]

plt.savefig(_static_path / "rbf-plot.png")
plt.clf()


# --8<-- [start:rbf-regr]
from sklearn.linear_model import LinearRegression

plt.figure(figsize=(17,3))
plt.plot(X["day"], y)
plt.plot(X["day"], LinearRegression().fit(Xt, y).predict(Xt), linewidth=2.0)
plt.title("pretty fly for a linear regression");
# --8<-- [end:rbf-regr]

plt.savefig(_static_path / "rbf-regr.png")
plt.clf()


######################################## Interval Encoder ###########################################
##########################################################################################

# --8<-- [start:interval-encoder-1]
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme()

xs = np.arange(0, 600)/100/np.pi
ys = np.sin(xs) + np.random.normal(0, 0.1, 600)

pred_ys = LinearRegression().fit(xs.reshape(-1, 1), ys).predict(xs.reshape(-1, 1))
plt.scatter(xs, ys)
plt.scatter(xs, pred_ys)
plt.title("not really the right pattern");
# --8<-- [end:interval-encoder-1]

plt.savefig(_static_path / "interval-encoder-1.png")
plt.clf()


# --8<-- [start:interval-encoder-2]
from sklego.preprocessing import IntervalEncoder

plt.figure(figsize = (16, 3))

for idx, sigma in enumerate([1, 0.1, 0.01, 0.001]):
    plt.subplot(140 + idx + 1)
    fs = IntervalEncoder(n_chunks=20, span=sigma, method='normal').fit(xs.reshape(-1, 1), ys)
    plt.scatter(xs, ys);
    plt.plot(xs, fs.transform(xs.reshape(-1, 1)), color='orange', linewidth=2.0)
    plt.title(f"span={sigma}");
# --8<-- [end:interval-encoder-2]

plt.savefig(_static_path / "interval-encoder-2.png")
plt.clf()

# --8<-- [start:interval-encoder-3]
from sklego.preprocessing import IntervalEncoder

plt.figure(figsize = (16, 3))

xs_extra = np.array([-1] + list(xs) + [3])
for idx, sigma in enumerate([1, 0.1, 0.01, 0.001]):
    plt.subplot(140 + idx + 1)
    fs = IntervalEncoder(n_chunks=20, span=sigma, method='normal').fit(xs.reshape(-1, 1), ys)
    plt.scatter(xs, ys);
    plt.plot(xs_extra, fs.transform(xs_extra.reshape(-1, 1)), color='orange', linewidth=2.0)
    plt.title(f"span={sigma}");
# --8<-- [end:interval-encoder-3]

plt.savefig(_static_path / "interval-encoder-3.png")
plt.clf()

######################################## Monotonic Interval Encoder ###########################################
##########################################################################################

# --8<-- [start:monotonic-1]
def generate_dataset(start, n=600):
    xs = np.arange(start, start + n)/100/np.pi
    y = np.sin(xs) + np.random.normal(0, 0.1, n)
    return xs.reshape(-1, 1), y
# --8<-- [end:monotonic-1]


# --8<-- [start:monotonic-2]
i = 0
plt.figure(figsize=(12, 6))
for method in ['average', 'normal']:
    for data_init in [50, 600, 1200, 2100]:
        i += 1
        X, y = generate_dataset(start=data_init)
        encoder = IntervalEncoder(n_chunks = 40, method=method, span=0.2)
        plt.subplot(240 + i)
        plt.title(f"method={method}")
        plt.scatter(X.reshape(-1), y);
        plt.plot(X.reshape(-1), encoder.fit_transform(X, y), color='orange', linewidth=2.0);
# --8<-- [end:monotonic-2]

plt.savefig(_static_path / "monotonic-2.png")
plt.clf()

# --8<-- [start:monotonic-3]
i = 0
plt.figure(figsize=(12, 6))
for method in ['increasing', 'decreasing']:
    for data_init in [50, 600, 1200, 2100]:
        i += 1
        X, y = generate_dataset(start=data_init)
        encoder = IntervalEncoder(n_chunks = 40, method=method, span=0.2)
        plt.subplot(240 + i)
        plt.title(f"method={method}")
        plt.scatter(X.reshape(-1), y);
        plt.plot(X.reshape(-1), encoder.fit_transform(X, y), color='orange', linewidth=2.0);
# --8<-- [end:monotonic-3]

plt.savefig(_static_path / "monotonic-3.png")
plt.clf()
