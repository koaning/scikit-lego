from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

#################################### Simulated Data ######################################
##########################################################################################

# --8<-- [start:simulated-data]
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set_theme()

n = 10000

def make_arr(mu1, mu2, std1=1, std2=1, p=0.5):
    res = np.where(np.random.uniform(0, 1, n) > p,
                    np.random.normal(mu1, std1, n),
                    np.random.normal(mu2, std2, n));
    return np.expand_dims(res, 1)

np.random.seed(42)
X1 = np.concatenate([make_arr(0, 4), make_arr(0, 4)], axis=1)
X2 = np.concatenate([make_arr(-3, 7), make_arr(2, 2)], axis=1)

plt.figure(figsize=(4,4))
plt.scatter(X1[:, 0], X1[:, 1], alpha=0.5)
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.5)
plt.title("simulated dataset");
# --8<-- [end:simulated-data]

plt.savefig(_static_path / "simulated-data.png")
plt.clf()

#################################### Model results #######################################
##########################################################################################

# --8<-- [start:model-results]
from sklego.naive_bayes import GaussianMixtureNB
cmap=sns.color_palette("flare", as_cmap=True)

X = np.concatenate([X1, X2])
y = np.concatenate([np.zeros(n), np.ones(n)])
plt.figure(figsize=(8, 8))
for i, k in enumerate([1, 2]):
    mod = GaussianMixtureNB(n_components=k).fit(X, y)
    plt.subplot(220 + i * 2 + 1)
    pred = mod.predict_proba(X)[:, 0]
    plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=cmap)
    plt.title(f"predict_proba k={k}")

    plt.subplot(220 + i * 2 + 2)
    pred = mod.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=cmap)
    plt.title(f"predict k={k}");
# --8<-- [end:model-results]

plt.savefig(_static_path / "model-results.png")
plt.clf()

#################################### Model density #######################################
##########################################################################################

# --8<-- [start:model-density]
gmm1 = mod.gmms_[0.0]
gmm2 = mod.gmms_[1.0]
plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.hist(gmm1[0].sample(n)[0], 30)
plt.title("model 1 - column 1 density")
plt.subplot(222)
plt.hist(gmm1[1].sample(n)[0], 30)
plt.title("model 1 - column 2 density")
plt.subplot(223)
plt.hist(gmm2[0].sample(n)[0], 30)
plt.title("model 2 - column 1 density")
plt.subplot(224)
plt.hist(gmm2[1].sample(n)[0], 30)
plt.title("model 2 - column 2 density");
# --8<-- [end:model-density]

plt.savefig(_static_path / "model-density.png")
plt.clf()