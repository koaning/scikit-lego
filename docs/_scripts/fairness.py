from pathlib import Path

# --8<-- [start:common-imports]
import matplotlib.pylab as plt
import seaborn as sns
sns.set_theme()
# --8<-- [end:common-imports]

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)


# --8<-- [start:predict-boston-simple]
import matplotlib.pylab as plt
import seaborn as sns
sns.set_theme()

from sklearn.datasets import load_boston
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X, y = load_boston(return_X_y=True)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])

plt.scatter(pipe.fit(X, y).predict(X), y)
plt.xlabel("predictions")
plt.ylabel("actual")
plt.title("plot that suggests it's not bad");
# --8<-- [end:predict-boston-simple]

plt.savefig(_static_path / "predict-boston-simple.png")
plt.clf()

# --8<-- [start:print-boston]
print(load_boston()["DESCR"][:1233])
# --8<-- [end:print-boston]

descr = load_boston()["DESCR"][:1233]
with open(_static_path / "boston-description.txt", "w") as f:
    f.write(descr)


# --8<-- [start:p-percent-score]
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklego.metrics import p_percent_score

sensitive_classification_dataset = pd.DataFrame({
    "x1": [1, 0, 1, 0, 1, 0, 1, 1],
    "x2": [0, 0, 0, 0, 0, 1, 1, 1],
    "y": [1, 1, 1, 0, 1, 0, 0, 0]}
)

X, y = sensitive_classification_dataset.drop(columns="y"), sensitive_classification_dataset["y"]
mod_unfair = LogisticRegression(solver="lbfgs").fit(X, y)

print("p_percent_score:", p_percent_score(sensitive_column="x2")(mod_unfair, X))
# --8<-- [end:p-percent-score]

# --8<-- [start:equal-opportunity-score]
import numpy as np
import pandas as pd
from sklego.metrics import equal_opportunity_score
import types

sensitive_classification_dataset = pd.DataFrame({
    "x1": [1, 0, 1, 0, 1, 0, 1, 1],
    "x2": [0, 0, 0, 0, 0, 1, 1, 1],
    "y": [1, 1, 1, 0, 1, 0, 0, 1]}
)

X, y = sensitive_classification_dataset.drop(columns="y"), sensitive_classification_dataset["y"]

mod_1 = types.SimpleNamespace()
mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 1, 1])
print("equal_opportunity_score:", equal_opportunity_score(sensitive_column="x2")(mod_1, X, y))

mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 1])
print("equal_opportunity_score:", equal_opportunity_score(sensitive_column="x2")(mod_1, X, y))

mod_1.predict = lambda X: np.array([1, 0, 1, 0, 1, 0, 0, 0])
print("equal_opportunity_score:", equal_opportunity_score(sensitive_column="x2")(mod_1, X, y))
# --8<-- [end:equal-opportunity-score]

# --8<-- [start:information-filter]
import pandas as pd
from sklearn.datasets import load_boston
from sklego.preprocessing import InformationFilter

X, y = load_boston(return_X_y=True)
df = pd.DataFrame(X,
    columns=["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]
)

X_drop = df.drop(columns=["lstat", "b"])

X_fair = InformationFilter(["lstat", "b"]).fit_transform(df)
X_fair = pd.DataFrame(X_fair, columns=[n for n in df.columns if n not in ["b", "lstat"]])


def simple_mod():
    """Create a simple model"""
    return Pipeline([("scale", StandardScaler()), ("mod", LinearRegression())])

base_mod = simple_mod().fit(X, y)
drop_mod = simple_mod().fit(X_drop, y)
fair_mod = simple_mod().fit(X_fair, y)

base_pred = base_mod.predict(X)
drop_pred = drop_mod.predict(X_drop)
fair_pred = fair_mod.predict(X_fair)
# --8<-- [end:information-filter]

# --8<-- [start:information-filter-coefs]
coefs = pd.DataFrame([
    base_mod.steps[1][1].coef_,
    drop_mod.steps[1][1].coef_,
    fair_mod.steps[1][1].coef_
    ],
    columns=df.columns)
coefs
# --8<-- [end:information-filter-coefs]

with open(_static_path / "information-filter-coefs.md", "w") as f:
    f.write(coefs.head().to_markdown(index=False))

# --8<-- [start:utils]
# We're using "lstat" to select the group to keep things simple
selector = df["lstat"] > np.quantile(df["lstat"], 0.5)

def bootstrap_means(preds, selector, n=2500, k=25):
    grp1 = np.random.choice(preds[selector], (n, k)).mean(axis=1)
    grp2 = np.random.choice(preds[~selector], (n, k)).mean(axis=1)
    return grp1 - grp2
# --8<-- [start:utils]


# --8<-- [start:original-situation]
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

sns.set_theme()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(base_pred, y)
plt.title(f"MSE: {mean_squared_error(y, base_pred)}")
plt.subplot(122)
plt.hist(bootstrap_means(base_pred, selector), bins=30, density=True, alpha=0.8)
plt.title(f"Fairness Proxy");
# --8<-- [end:original-situation]
plt.savefig(_static_path / "original-situation.png")
plt.clf()

# --8<-- [start:drop-two]
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

sns.set_theme()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(drop_pred, y)
plt.title(f"MSE: {mean_squared_error(y, drop_pred)}")
plt.subplot(122)
plt.hist(bootstrap_means(base_pred, selector), bins=30, density=True, alpha=0.8)
plt.hist(bootstrap_means(drop_pred, selector), bins=30, density=True, alpha=0.8)
plt.title(f"Fairness Proxy");
# --8<-- [end:drop-two]

plt.savefig(_static_path / "drop-two.png")
plt.clf()


# --8<-- [start:use-info-filter]
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

sns.set_theme()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(fair_pred, y)
plt.title(f"MSE: {mean_squared_error(y, fair_pred)}")
plt.subplot(122)
plt.hist(bootstrap_means(base_pred, selector), bins=30, density=True, alpha=0.8)
plt.hist(bootstrap_means(fair_pred, selector), bins=30, density=True, alpha=0.8)
plt.title(f"Fairness Proxy");
# --8<-- [end:use-info-filter]

plt.savefig(_static_path / "use-info-filter.png")
plt.clf()

# --8<-- [start:demographic-parity]
from sklego.linear_model import DemographicParityClassifier
from sklearn.linear_model import LogisticRegression
from sklego.metrics import p_percent_score

df_clf = df.assign(lstat=lambda d: d["lstat"] > np.median(d["lstat"]))
y_clf = y > np.median(y)

normal_classifier = LogisticRegression(solver="lbfgs")
_ = normal_classifier.fit(df_clf, y_clf)
fair_classifier = DemographicParityClassifier(sensitive_cols="lstat", covariance_threshold=0.5)
_ = fair_classifier.fit(df_clf, y_clf)
# --8<-- [end:demographic-parity]

# --8<-- [start:demographic-parity-grid]
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
fair_classifier = GridSearchCV(
    estimator=DemographicParityClassifier(sensitive_cols="lstat", covariance_threshold=0.5),
    param_grid={"estimator__covariance_threshold": np.linspace(0.01, 1.00, 20)},
    cv=5,
    refit="accuracy_score",
    return_train_score=True,
    scoring={
        "p_percent_score": p_percent_score("lstat"),
        "accuracy_score": make_scorer(accuracy_score)
    }
)

fair_classifier.fit(df_clf, y_clf)

pltr = (pd.DataFrame(fair_classifier.cv_results_)
        .set_index("param_estimator__covariance_threshold"))

p_score = p_percent_score("lstat")(normal_classifier, df_clf, y_clf)
acc_score = accuracy_score(normal_classifier.predict(df_clf), y_clf)
# --8<-- [end:demographic-parity-grid]


# --8<-- [start:demographic-parity-grid-results]
import matplotlib.pylab as plt
import seaborn as sns

sns.set_theme()

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.plot(np.array(pltr.index), pltr["mean_test_p_percent_score"], label="fairclassifier")
plt.plot(np.linspace(0, 1, 2), [p_score for _ in range(2)], label="logistic-regression")
plt.xlabel("covariance threshold")
plt.legend()
plt.title("p% score")
plt.subplot(122)
plt.plot(np.array(pltr.index), pltr["mean_test_accuracy_score"], label="fairclassifier")
plt.plot(np.linspace(0, 1, 2), [acc_score for _ in range(2)], label="logistic-regression")
plt.xlabel("covariance threshold")
plt.legend()
plt.title("accuracy");
# --8<-- [end:demographic-parity-grid-results]

plt.savefig(_static_path / "demographic-parity-grid-results.png")
plt.clf()

# --8<-- [start:equal-opportunity-grid]
from sklego.linear_model import EqualOpportunityClassifier

fair_classifier = GridSearchCV(
    estimator=EqualOpportunityClassifier(
        sensitive_cols="lstat", 
        covariance_threshold=0.5,
        positive_target=True,
    ),
    param_grid={"estimator__covariance_threshold": np.linspace(0.001, 1.00, 20)},
    cv=5,
    n_jobs=-1,
    refit="accuracy_score",
    return_train_score=True,
    scoring={
        "p_percent_score": p_percent_score("lstat"),
        "equal_opportunity_score": equal_opportunity_score("lstat"),
        "accuracy_score": make_scorer(accuracy_score)
    }
)

fair_classifier.fit(df_clf, y_clf)

pltr = (pd.DataFrame(fair_classifier.cv_results_)
        .set_index("param_estimator__covariance_threshold"))

p_score = p_percent_score("lstat")(normal_classifier, df_clf, y_clf)
acc_score = accuracy_score(normal_classifier.predict(df_clf), y_clf)
# --8<-- [end:equal-opportunity-grid]

# --8<-- [start:equal-opportunity-grid-results]
import matplotlib.pylab as plt
import seaborn as sns

sns.set_theme()

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.plot(np.array(pltr.index), pltr["mean_test_equal_opportunity_score"], label="fairclassifier")
plt.plot(np.linspace(0, 1, 2), [p_score for _ in range(2)], label="logistic-regression")
plt.xlabel("covariance threshold")
plt.legend()
plt.title("equal opportunity score")
plt.subplot(122)
plt.plot(np.array(pltr.index), pltr["mean_test_accuracy_score"], label="fairclassifier")
plt.plot(np.linspace(0, 1, 2), [acc_score for _ in range(2)], label="logistic-regression")
plt.xlabel("covariance threshold")
plt.legend()
plt.title("accuracy");

# --8<-- [end:equal-opportunity-grid-results]

plt.savefig(_static_path / "equal-opportunity-grid-results.png")
plt.clf()