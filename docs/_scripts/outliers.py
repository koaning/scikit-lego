from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

######################################## Setup ###########################################
##########################################################################################

# --8<-- [start:setup]
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.datasets import load_iris
from pandas.plotting import parallel_coordinates
sns.set_theme()

X_orig, y = load_iris(return_X_y=True, as_frame=True)

def plot_model(mod, components, threshold):
    mod = mod(n_components=components, threshold=threshold, random_state=111).fit(X_orig)
    X = X_orig.copy()
    X['label'] = mod.predict(X)

    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    parallel_coordinates(X.loc[lambda d: d['label'] == 1], class_column='label', alpha=0.5)
    parallel_coordinates(X.loc[lambda d: d['label'] == -1], class_column='label', color='red', alpha=0.7)
    plt.title("outlier shown via parallel coordinates")

    if components == 2:
        plt.subplot(122)
        X_reduced = mod.transform(X_orig)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X['label'], cmap="coolwarm_r")
        plt.title("outlier shown in 2d");
# --8<-- [end:setup]


######################################### PCA ############################################
##########################################################################################

# --8<-- [start:pca-outlier]
from sklego.decomposition import PCAOutlierDetection
plot_model(PCAOutlierDetection, components=2, threshold=0.1)
# --8<-- [end:pca-outlier]

plt.savefig(_static_path / "pca-outlier.png")
plt.clf()


######################################### UMAP ###########################################
##########################################################################################

# --8<-- [start:umap-outlier]
from sklego.decomposition import UMAPOutlierDetection
plot_model(UMAPOutlierDetection, components=2, threshold=0.1)
# --8<-- [end:umap-outlier]

plt.savefig(_static_path / "umap-outlier.png")
plt.clf()

######################################### GMMOutlierDetector ###########################################
##########################################################################################

# --8<-- [start:gmm-outlier]
from sklego.mixture import GMMOutlierDetector

mod = GMMOutlierDetector(n_components=4, threshold=0.99).fit(X_orig)
X = X_orig.copy()
X['label'] = mod.predict(X)

plt.figure(figsize=(12, 3))
parallel_coordinates(X.loc[lambda d: d['label'] == 1], class_column='label', alpha=0.5)
parallel_coordinates(X.loc[lambda d: d['label'] == -1], class_column='label', color='red', alpha=0.7)
plt.title("outlier shown via parallel coordinates");
# --8<-- [end:gmm-outlier]

plt.savefig(_static_path / "gmm-outlier.png")
plt.clf()


######################################### BayesianGMMOutlierDetector ###########################################
##########################################################################################

# --8<-- [start:bayesian-gmm-outlier]
from sklego.mixture import BayesianGMMOutlierDetector

mod = BayesianGMMOutlierDetector(n_components=4, threshold=0.99).fit(X_orig)
X = X_orig.copy()
X['label'] = mod.predict(X)

plt.figure(figsize=(12, 3))
parallel_coordinates(X.loc[lambda d: d['label'] == 1], class_column='label', alpha=0.5)
parallel_coordinates(X.loc[lambda d: d['label'] == -1], class_column='label', color='red', alpha=0.7)
plt.title("outlier shown via parallel coordinates");
# --8<-- [end:bayesian-gmm-outlier]

plt.savefig(_static_path / "bayesian-gmm-outlier.png")
plt.clf()


######################################### BayesianGMMOutlierDetector ###########################################
##########################################################################################

# --8<-- [start:regr-outlier]
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklego.meta import RegressionOutlierDetector

sns.set_theme()

# generate random data for illustrative example
np.random.seed(42)
X = np.random.normal(0, 1, (100, 1))
y = 1 + np.sum(X, axis=1).reshape(-1, 1) + np.random.normal(0, 0.2, (100, 1))
for i in [20, 25, 50, 80]:
    y[i] += 2
X = np.concatenate([X, y], axis=1)

# fit and plot
mod = RegressionOutlierDetector(LinearRegression(), column=1)
mod.fit(X)
plt.scatter(X[:, 0], X[:, 1], c=mod.predict(X), cmap='coolwarm_r');
# --8<-- [end:regr-outlier]

plt.savefig(_static_path / "regr-outlier.png")
plt.clf()
