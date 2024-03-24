from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

######################################## TimeGapSplit ###########################################
##########################################################################################

# --8<-- [start:setup]
from datetime import timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklego.model_selection import TimeGapSplit

# Plotting helper function
def plot_cv(cv, X):
    """Plot all the folds on time axis"""
    X_index_df = cv._join_date_and_x(X)

    plt.figure(figsize=(16, 4))
    for i, split in enumerate(cv.split(X)):
        x_idx, y_idx = split
        x_dates = X_index_df.iloc[x_idx]["__date__"].unique()
        y_dates = X_index_df.iloc[y_idx]["__date__"].unique()
        plt.plot(x_dates, i*np.ones(x_dates.shape), c="steelblue")
        plt.plot(y_dates, i*np.ones(y_dates.shape), c="orange")

    plt.legend(("training", "validation"), loc="upper left")
    plt.ylabel("Fold id")
    plt.axvline(x=X_index_df["__date__"].min(), color="gray", label="x")
    plt.axvline(x=X_index_df["__date__"].max(), color="gray", label="d")

# Random data creation
df = (pd.DataFrame(np.random.randint(0, 30, size=(30, 4)), columns=list("ABCy"))
      .assign(date=pd.date_range(start="1/1/2018", end="1/30/2018")[::-1]))

print(df.shape)
# (30, 5)

print(df.head())
# --8<-- [end:setup]

with open(_static_path / "ts.md", "w") as f:
    f.write(df.head().to_markdown(index=False))

# --8<-- [start:example-1]
cv = TimeGapSplit(
    date_serie=df["date"],
    train_duration=timedelta(days=10),
    valid_duration=timedelta(days=2),
    gap_duration=timedelta(days=1)
)

plot_cv(cv, df)
# --8<-- [end:example-1]

plt.savefig(_static_path / "example-1.png")
plt.clf()



# --8<-- [start:example-2]
cv = TimeGapSplit(
    date_serie=df["date"],
    train_duration=timedelta(days=10),
    valid_duration=timedelta(days=5),
    gap_duration=timedelta(days=1)
)

plot_cv(cv, df)
# --8<-- [end:example-2]

plt.savefig(_static_path / "example-2.png")
plt.clf()


# --8<-- [start:example-3]
cv = TimeGapSplit(
    date_serie=df["date"],
    train_duration=timedelta(days=10),
    valid_duration=timedelta(days=2),
    gap_duration=timedelta(days=1),
    window="expanding"
)

plot_cv(cv, df)
# --8<-- [end:example-3]

plt.savefig(_static_path / "example-3.png")
plt.clf()


# --8<-- [start:example-4]
cv = TimeGapSplit(
    date_serie=df["date"],
    train_duration=None,
    valid_duration=timedelta(days=3),
    gap_duration=timedelta(days=2),
    n_splits=3
)

plot_cv(cv, df)
# --8<-- [end:example-4]

plt.savefig(_static_path / "example-4.png")
plt.clf()


# --8<-- [start:example-5]
cv = TimeGapSplit(
    date_serie=df["date"],
    train_duration=timedelta(days=10),
    valid_duration=timedelta(days=2),
    gap_duration=timedelta(days=1),
    n_splits=4
)

plot_cv(cv, df)
# --8<-- [end:example-5]

plt.savefig(_static_path / "example-5.png")
plt.clf()

# --8<-- [start:summary]
cv.summary(df)
# --8<-- [end:summary]

with open(_static_path / "summary.md", "w") as f:
    f.write(cv.summary(df).to_markdown(index=False))


######################################## GroupedTSSplit ###########################################
##########################################################################################

# --8<-- [start:grp-setup]
import numpy as np
import pandas as pd

X = np.random.randint(low=1, high=1000, size=17)
y = np.random.randint(low=1, high=1000, size=17)
groups = np.array([2000,2000,2000,2001,2002,2002,2003,2004,2004,2004,2004,2004,2005,2005,2006,2006,2007])

df = pd.DataFrame(np.vstack((X,y)).T, index=groups, columns=['X','y'])
df.head(10)
# --8<-- [end:grp-setup]

with open(_static_path / "grp-ts.md", "w") as f:
    f.write(df.head(10).to_markdown(index=False))

# --8<-- [start:grp-ts-split]
from sklego.model_selection import GroupTimeSeriesSplit
cv = GroupTimeSeriesSplit(n_splits=3)

def print_folds(cv, X, y, groups):
    for kfold, (train, test) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {kfold+1}:")
        print(f"Train = {df.iloc[train].index.tolist()}")
        print(f"Test = {df.iloc[test].index.tolist()}\n\n")

print_folds(cv, X, y, groups)
# --8<-- [end:grp-ts-split]


# --8<-- [start:grp-summary]
cv.summary()
# --8<-- [end:grp-summary]

with open(_static_path / "grp-summary.md", "w") as f:
    f.write(cv.summary().to_markdown(index=False))


# --8<-- [start:grid-search]
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# reshape X for the Lasso model
X = X.reshape(-1,1)

# initiate the cross validator
cv = GroupTimeSeriesSplit(n_splits=3)

# generate the train-test splits
cv_splits = cv.split(X=X, y=y, groups=groups)

# initiate the Lasso model
Lasso(random_state=0, tol=0.1, alpha=0.8).fit(X, y, groups)
pipe = Pipeline([("reg", Lasso(random_state=0, tol=0.1))])


# initiate GridSearchCv with cv_splits as parameter
alphas = [0.1, 0.5, 0.8]
grid = GridSearchCV(pipe, {"reg__alpha": alphas}, cv=cv_splits)
grid.fit(X, y)
grid.best_estimator_.get_params()["reg__alpha"]
# 0.8
# --8<-- [end:grid-search]



######################################## ClusterKfold ####################################
##########################################################################################

# --8<-- [start:cluster-fold-start]
from sklego.model_selection import ClusterFoldValidation
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=5, random_state=42)
folder = ClusterFoldValidation(clusterer)
# --8<-- [end:cluster-fold-start]


# --8<-- [start:cluster-fold-plot]
import matplotlib.pylab as plt
import numpy as np

X_orig = np.random.uniform(0, 1, (1000, 2))
for i, split in enumerate(folder.split(X_orig)):
    x_train, x_valid = split
    plt.scatter(X_orig[x_valid, 0], X_orig[x_valid, 1], label=f"split {i}")
plt.legend();
# --8<-- [end:cluster-fold-plot]
