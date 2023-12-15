from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

#################################### GMMClassifier #######################################
##########################################################################################

# --8<-- [start:gmm-classifier]
import numpy as np
import matplotlib.pylab as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from sklego.mixture import GMMClassifier

n = 1000
X, y = make_moons(n)
X = X + np.random.normal(n, 0.12, (n, 2))
X = StandardScaler().fit_transform(X)
U = np.random.uniform(-2, 2, (10000, 2))

mod = GMMClassifier(n_components=4).fit(X, y)

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=mod.predict(X), s=8)
plt.title("classes of points");

plt.subplot(122)
plt.scatter(U[:, 0], U[:, 1], c=mod.predict_proba(U)[:, 1], s=8)
plt.title("classifier boundary");
# --8<-- [end:gmm-classifier]

plt.savefig(_static_path / "gmm-classifier.png")
plt.clf()


################################## GMMOutlierDetector ####################################
##########################################################################################

# --8<-- [start:gmm-outlier-detector]
import numpy as np
import matplotlib.pylab as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from sklego.mixture import GMMOutlierDetector

n = 1000
X = make_moons(n)[0] + np.random.normal(n, 0.12, (n, 2))
X = StandardScaler().fit_transform(X)
U = np.random.uniform(-2, 2, (10000, 2))

mod = GMMOutlierDetector(n_components=16, threshold=0.95).fit(X)

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=mod.score_samples(X), s=8)
plt.title("likelihood of points given mixture of 16 gaussians");

plt.subplot(122)
plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)
plt.title("outlier selection");
# --8<-- [end:gmm-outlier-detector]

plt.savefig(_static_path / "gmm-outlier-detector.png")
plt.clf()

################################# Different Thresholds ###################################
##########################################################################################

# --8<-- [start:gmm-outlier-multi-threshold]
plt.figure(figsize=(14, 5))
for i in range(1, 5):
    mod = GMMOutlierDetector(n_components=16, threshold=i, method="stddev").fit(X)
    plt.subplot(140 + i)
    plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)
    plt.title(f"outlier sigma={i}");
# --8<-- [end:gmm-outlier-multi-threshold]

plt.savefig(_static_path / "gmm-outlier-multi-threshold.png")
plt.clf()

########################################### KDE ##########################################
##########################################################################################

# --8<-- [start:outlier-mixture-threshold]
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set_theme()

score_samples = np.random.beta(220, 10, 3000)
density = gaussian_kde(score_samples)
likelihood_range = np.linspace(0.80, 1.0, 10000)

index_max_y = np.argmax(density(likelihood_range))
mean_likelihood = likelihood_range[index_max_y]
new_likelihoods = score_samples[score_samples < mean_likelihood]
new_likelihoods_std = np.sqrt(np.sum((new_likelihoods - mean_likelihood) ** 2) / (len(new_likelihoods) - 1))

plt.figure(figsize=(14, 3))
plt.subplot(121)
plt.plot(likelihood_range, density(likelihood_range), "k")
xs = np.linspace(0.8, 1.0, 2000)
plt.fill_between(xs, density(xs), alpha=0.8)
plt.title("log-lik values from with GMM, quantile is based on blue part")

plt.subplot(122)
plt.plot(likelihood_range, density(likelihood_range), "k")
plt.vlines(mean_likelihood, 0, density(mean_likelihood), linestyles="dashed")
xs = np.linspace(0.8, mean_likelihood, 2000)
plt.fill_between(xs, density(xs), alpha=0.8)
plt.title("log-lik values from with GMM, stddev is based on blue part")
# --8<-- [end:outlier-mixture-threshold]

plt.savefig(_static_path / "outlier-mixture-threshold.png")
plt.clf()
