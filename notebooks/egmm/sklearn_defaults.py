# ruff: noqa: T201, C408, B023
import time
import warnings

import numpy as np
import polars as pl
from egmm import egmm_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
X = np.load("glove100.npy").astype(np.float64)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
X = X[np.random.default_rng(0).permutation(len(X))]
Xtr, Xte = X[:320_000], X[320_000:]
rows = []
for k in [256, 1024]:
    tic = time.perf_counter()
    m = KMeans(k, random_state=0).fit(Xtr.astype(np.float32))
    sec = time.perf_counter() - tic
    mse = lambda A: float((pairwise_distances_argmin_min(A.astype(np.float32), m.cluster_centers_)[1] ** 2).mean())
    rows.append(
        dict(
            kind="kmeans",
            dataset=f"GloVe-100 normalized (400k x 100, k={k})",
            method="sklearn KMeans() defaults",
            iters=int(m.n_iter_),
            seconds=sec,
            test=mse(Xte),
            train=mse(Xtr),
        )
    )
    print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv("results/sklearn_defaults.csv")
tic = time.perf_counter()
g = GaussianMixture(256, covariance_type="diag", random_state=0).fit(Xtr)
sec = time.perf_counter() - tic
p = (g.weights_, g.means_, g.covariances_)
rows.append(
    dict(
        kind="gmm",
        dataset="GloVe-100 normalized (320k train / 80k test, k=256)",
        method="sklearn GaussianMixture() defaults",
        iters=int(g.n_iter_),
        seconds=sec,
        test=egmm_score(Xte, *p),
        train=egmm_score(Xtr, *p),
    )
)
print(rows[-1], flush=True)
pl.DataFrame(rows).write_csv("results/sklearn_defaults.csv")
print("DONE", flush=True)
