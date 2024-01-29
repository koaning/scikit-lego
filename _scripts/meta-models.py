from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)


#################################### Thresholder #########################################
##########################################################################################
# --8<-- [start:skewed-data]
import matplotlib.pylab as plt
import seaborn as sns

sns.set_theme()
cmap=sns.color_palette("flare", as_cmap=True)

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, make_scorer

from sklego.meta import Thresholder

X, y = make_blobs(1000, centers=[(0, 0), (1.5, 1.5)], cluster_std=[1, 0.5])
plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap=cmap);
# --8<-- [end:skewed-data]

plt.savefig(_static_path / "skewed-data.png")
plt.clf()


# --8<-- [start:cross-validation]
# %%time

pipe = Pipeline([
    ("model", Thresholder(LogisticRegression(solver="lbfgs"), threshold=0.1))
])

mod = GridSearchCV(
    estimator=pipe,
    param_grid={"model__threshold": np.linspace(0.1, 0.9, 500)},
    scoring={
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "accuracy": make_scorer(accuracy_score)
    },
    refit="precision",
    cv=5
)

_ = mod.fit(X, y)
# --8<-- [end:cross-validation]

# --8<-- [start:threshold-chart]
(pd.DataFrame(mod.cv_results_)
 .set_index("param_model__threshold")
 [["mean_test_precision", "mean_test_recall", "mean_test_accuracy"]]
 .plot(figsize=(16, 6)));
# --8<-- [end:threshold-chart]

plt.savefig(_static_path / "threshold-chart.png")
plt.clf()

# --8<-- [start:cross-validation-no-refit]
# %%time

# Train an original model
orig_model = LogisticRegression(solver="lbfgs")
orig_model.fit(X, y)

# Ensure that refit=False
pipe = Pipeline([
    ("model", Thresholder(orig_model, threshold=0.1, refit=False))
])

# This should now be a fair bit quicker.
mod = GridSearchCV(
    estimator=pipe,
    param_grid = {"model__threshold": np.linspace(0.1, 0.9, 50)},
    scoring={
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "accuracy": make_scorer(accuracy_score)
    },
    refit="precision",
    cv=5
)

_ = mod.fit(X, y);
# --8<-- [end:cross-validation-no-refit]

################################## Grouped Predictor #####################################
##########################################################################################

# --8<-- [start:grouped-predictor-setup]
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklego.datasets import load_chicken
from sklego.preprocessing import ColumnSelector

def plot_model(model):
    df = load_chicken(as_frame=True)

    _ = model.fit(df[["diet", "time"]], df["weight"])
    metric_df = (df[["diet", "time", "weight"]]
        .assign(pred=lambda d: model.predict(d[["diet", "time"]]))
    )

    metric = mean_absolute_error(metric_df["weight"], metric_df["pred"])

    plt.figure(figsize=(12, 4))
    plt.scatter(df["time"], df["weight"])
    for i in [1, 2, 3, 4]:
        pltr = metric_df[["time", "diet", "pred"]].drop_duplicates().loc[lambda d: d["diet"] == i]
        plt.plot(pltr["time"], pltr["pred"], color=".rbgy"[i])
    plt.title(f"linear model per group, MAE: {np.round(metric, 2)}");
# --8<-- [end:grouped-predictor-setup]

# --8<-- [start:baseline-model]
feature_pipeline = Pipeline([
    ("datagrab", FeatureUnion([
         ("discrete", Pipeline([
             ("grab", ColumnSelector("diet")),
             ("encode", OneHotEncoder(categories="auto", sparse=False))
         ])),
         ("continuous", Pipeline([
             ("grab", ColumnSelector("time")),
             ("standardize", StandardScaler())
         ]))
    ]))
])

pipe = Pipeline([
    ("transform", feature_pipeline),
    ("model", LinearRegression())
])

plot_model(pipe)
# --8<-- [end:baseline-model]

plt.savefig(_static_path / "baseline-model.png")
plt.clf()

# --8<-- [start:grouped-model]
from sklego.meta import GroupedPredictor
mod = GroupedPredictor(LinearRegression(), groups=["diet"])
plot_model(mod)
# --8<-- [end:grouped-model]

plt.savefig(_static_path / "grouped-model.png")
plt.clf()

# --8<-- [start:grouped-dummy-model]
from sklearn.dummy import DummyRegressor

feature_pipeline = Pipeline([
    ("datagrab", FeatureUnion([
         ("discrete", Pipeline([
             ("grab", ColumnSelector("diet")),
         ])),
         ("continuous", Pipeline([
             ("grab", ColumnSelector("time")),
         ]))
    ]))
])

pipe = Pipeline([
    ("transform", feature_pipeline),
    ("model", GroupedPredictor(DummyRegressor(strategy="mean"), groups=[0, 1]))
])

plot_model(pipe)
# --8<-- [end:grouped-dummy-model]

plt.savefig(_static_path / "grouped-dummy-model.png")
plt.clf()

################################# Grouped Transformer ####################################
##########################################################################################

# --8<-- [start:penguins]
from sklego.datasets import load_penguins

df_penguins = (
    load_penguins(as_frame=True)
    .dropna()
    .drop(columns=["island", "bill_depth_mm", "bill_length_mm", "species"])
)

df_penguins.head()
# --8<-- [end:penguins]

with open(_static_path / "penguins.md", "w") as f:
    f.write(df_penguins.head().to_markdown(index=False))

# --8<-- [start:grouped-transform]
from sklearn.preprocessing import StandardScaler
from sklego.meta import GroupedTransformer

X = df_penguins.drop(columns=["sex"]).values

X_tfm = StandardScaler().fit_transform(X)
X_tfm_grp = (GroupedTransformer(
    transformer=StandardScaler(),
    groups=["sex"]
    )
    .fit_transform(df_penguins)
)
# --8<-- [end:grouped-transform]

# --8<-- [start:grouped-transform-plot]
import matplotlib.pylab as plt
import seaborn as sns
sns.set_theme()

plt.figure(figsize=(12, 6))
plt.subplot(121)

plt.scatter(X_tfm[:, 0], X_tfm[:, 1], c=df_penguins["sex"] == "male", cmap=cmap)
plt.xlabel("norm flipper len")
plt.ylabel("norm body mass")
plt.title("scaled data, not normalised by gender")

plt.subplot(122)
plt.scatter(X_tfm_grp[:, 0], X_tfm_grp[:, 1], c=df_penguins["sex"] == "male", cmap=cmap)
plt.xlabel("norm flipper len")
plt.ylabel("norm body mass")
plt.title("scaled data *and* normalised by gender");
# --8<-- [end:grouped-transform-plot]

plt.savefig(_static_path / "grouped-transform.png")
plt.clf()

# --8<-- [start:ts-data]
from sklego.datasets import make_simpleseries

yt = make_simpleseries(seed=1)
df = (pd.DataFrame({"yt": yt,
                   "date": pd.date_range("2000-01-01", periods=len(yt))})
      .assign(m=lambda d: d.date.dt.month)
      .reset_index())

plt.figure(figsize=(12, 3))
plt.plot(make_simpleseries(seed=1));

# --8<-- [end:ts-data]
plt.savefig(_static_path / "ts-data.png")
plt.clf()

# --8<-- [start:decay-model]
from sklearn.dummy import DummyRegressor
from sklego.meta import GroupedPredictor, DecayEstimator

mod1 = (GroupedPredictor(DummyRegressor(), groups=["m"])
        .fit(df[["m"]], df["yt"]))

mod2 = (GroupedPredictor(DecayEstimator(DummyRegressor(), decay_func="exponential", decay_rate=0.9), groups=["m"])
        .fit(df[["index", "m"]], df["yt"]))

plt.figure(figsize=(12, 3))
plt.plot(df["yt"], alpha=0.5);
plt.plot(mod1.predict(df[["m"]]), label="grouped")
plt.plot(mod2.predict(df[["index", "m"]]), label="decayed")
plt.legend();
# --8<-- [end:decay-model]

plt.savefig(_static_path / "decay-model.png")
plt.clf()


# --8<-- [start:decay-functions]
from sklego.meta._decay_utils import exponential_decay, linear_decay, sigmoid_decay, stepwise_decay

fig = plt.figure(figsize=(12, 6))

for i, name, func, kwargs in zip(
    range(1, 5),
    ("exponential", "linear", "sigmoid", "stepwise"),
    (exponential_decay, linear_decay, sigmoid_decay, stepwise_decay),
    ({"decay_rate": 0.995}, {"min_value": 0.1}, {}, {"n_steps": 8})
    ):

    ax = fig.add_subplot(2, 2, i)
    x, y = None, np.arange(1000)
    ax.plot(func(x,y, **kwargs))
    ax.set_title(f'decay_func="{name}"')

plt.tight_layout()
# --8<-- [end:decay-functions]

plt.savefig(_static_path / "decay-functions.png")
plt.clf()

# --8<-- [start:make-blobs]
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set_theme()
cmap=sns.color_palette("flare", as_cmap=True)
np.random.seed(42)

n1, n2, n3 = 100, 500, 50
X = np.concatenate([np.random.normal(0, 1, (n1, 2)),
                    np.random.normal(2, 1, (n2, 2)),
                    np.random.normal(3, 1, (n3, 2))],
                   axis=0)
y = np.concatenate([np.zeros((n1, 1)),
                    np.ones((n2, 1)),
                    np.zeros((n3, 1))],
                   axis=0).reshape(-1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap);
# --8<-- [end:make-blobs]

plt.savefig(_static_path / "make-blobs.png")
plt.clf()

# --8<-- [start:simple-classifier]
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

mod = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=10000)
cfm = confusion_matrix(y, mod.fit(X, y).predict(X))
cfm
# array([[ 72,  78],
#        [  4, 496]])

# --8<-- [end:simple-classifier]

# --8<-- [start:normalized-cfm]
cfm.T / cfm.T.sum(axis=1).reshape(-1, 1)
# array([[0.94736842, 0.05263158],
#        [0.1358885 , 0.8641115 ]])
# --8<-- [end:normalized-cfm]

# --8<-- [start:help-functions]
def false_positives(mod, x, y):
    return (mod.predict(x) != y)[y == 1].sum()

def false_negatives(mod, x, y):
    return (mod.predict(x) != y)[y == 0].sum()
# --8<-- [end:help-functions]

# --8<-- [start:confusion-balancer]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklego.meta import ConfusionBalancer

cf_mod = ConfusionBalancer(LogisticRegression(solver="lbfgs", max_iter=1000), alpha=1.0)

grid = GridSearchCV(
    cf_mod,
    param_grid={"alpha": np.linspace(-1.0, 3.0, 31)},
    scoring={
        "accuracy": make_scorer(accuracy_score),
        "positives": false_positives,
        "negatives": false_negatives
    },
    n_jobs=-1,
    return_train_score=True,
    refit="negatives",
    cv=5
)
grid
# --8<-- [end:confusion-balancer]

from sklearn.utils import estimator_html_repr
with open(_static_path / "confusion-balanced-grid.html", "w") as f:
    f.write(estimator_html_repr(grid))



# --8<-- [start:confusion-balancer-results]
df = pd.DataFrame(grid.fit(X, y).cv_results_)
plt.figure(figsize=(12, 3))

plt.subplot(121)
plt.plot(df["param_alpha"], df["mean_test_positives"], label="false positives")
plt.plot(df["param_alpha"], df["mean_test_negatives"], label="false negatives")
plt.legend()
plt.subplot(122)
plt.plot(df["param_alpha"], df["mean_test_accuracy"], label="test accuracy")
plt.plot(df["param_alpha"], df["mean_train_accuracy"], label="train accuracy")
plt.legend();
# --8<-- [end:confusion-balancer-results]

plt.savefig(_static_path / "confusion-balancer-results.png")
plt.clf()

# --8<-- [start:zero-inflated]
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklego.meta import ZeroInflatedRegressor

np.random.seed(0)
X = np.random.randn(10000, 4)
y = ((X[:, 0]>0) & (X[:, 1]>0)) * np.abs(X[:, 2] * X[:, 3]**2) # many zeroes here, in about 75% of the cases.

zir = ZeroInflatedRegressor(
    classifier=RandomForestClassifier(random_state=0),
    regressor=RandomForestRegressor(random_state=0)
)

print("ZIR (RFC+RFR) r²:", cross_val_score(zir, X, y).mean())
print("RFR r²:", cross_val_score(RandomForestRegressor(random_state=0), X, y).mean())
# --8<-- [end:zero-inflated]

# --8<-- [start:outlier-classifier]
import numpy as np
from sklego.meta.outlier_classifier import OutlierClassifier
from sklearn.ensemble import IsolationForest

n_normal = 10_000
n_outlier = 100
np.random.seed(0)
X = np.hstack((np.random.normal(size=n_normal), np.random.normal(10, size=n_outlier))).reshape(-1,1)
y = np.hstack((np.asarray([0]*n_normal), np.asarray([1]*n_outlier)))

clf = OutlierClassifier(IsolationForest(n_estimators=1000, contamination=n_outlier/n_normal, random_state=0))
clf.fit(X, y)
# --8<-- [end:outlier-classifier]

from sklearn.utils import estimator_html_repr
with open(_static_path / "outlier-classifier.html", "w") as f:
    f.write(estimator_html_repr(clf))

# --8<-- [start:outlier-classifier-output]
print("inlier: ", clf.predict([[0]]))
print("outlier: ", clf.predict([[10]]))
# --8<-- [end:outlier-classifier-output]

# inlier:  [0.]
# outlier:  [1.]

# --8<-- [start:outlier-classifier-proba]
clf.predict_proba([[10]])
# --8<-- [end:outlier-classifier-proba]

# array([[0.0376881, 0.9623119]])

# --8<-- [start:outlier-classifier-stacking]
from sklearn.ensemble import StackingClassifier, RandomForestClassifier

estimators = [
    ("anomaly", OutlierClassifier(IsolationForest())),
    ("classifier", RandomForestClassifier())
    ]

stacker = StackingClassifier(estimators, stack_method="predict_proba", passthrough=True)
stacker.fit(X,y)
# --8<-- [end:outlier-classifier-stacking]

from sklearn.utils import estimator_html_repr
with open(_static_path / "outlier-classifier-stacking.html", "w") as f:
    f.write(estimator_html_repr(stacker))

# --8<-- [start:ordinal-classifier-data]
import pandas as pd

url = "https://stats.idre.ucla.edu/stat/data/ologit.dta"
df = pd.read_stata(url).assign(apply_codes = lambda t: t["apply"].cat.codes)

target = "apply_codes"
features = [c for c in df.columns if c not in {target, "apply"}]

X, y = df[features].to_numpy(), df[target].to_numpy()
df.head()
# --8<-- [end:ordinal-classifier-data]

with open(_static_path / "ordinal_data.md", "w") as f:
    f.write(df.head().to_markdown(index=False))

# --8<-- [start:ordinal-classifier]
from sklearn.linear_model import LogisticRegression
from sklego.meta import OrdinalClassifier

ord_clf = OrdinalClassifier(LogisticRegression(), n_jobs=-1, use_calibration=False)
_ = ord_clf.fit(X, y)
ord_clf.predict_proba(X[0])
# --8<-- [end:ordinal-classifier]

print(ord_clf.predict_proba(X[0]))

# --8<-- [start:ordinal-classifier-with-calibration]
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklego.meta import OrdinalClassifier

calibration_kwargs = {...}

ord_clf = OrdinalClassifier(
    estimator=LogisticRegression(),
    use_calibration=True,
    calibration_kwargs=calibration_kwargs
)

# This is equivalent to:
estimator = CalibratedClassifierCV(LogisticRegression(), **calibration_kwargs)
ord_clf = OrdinalClassifier(estimator)
# --8<-- [end:ordinal-classifier-with-calibration]
