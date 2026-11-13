# ruff: noqa: T201, C408
import time
import warnings

import numpy as np
import polars as pl
from egmm import egmm, egmm_score
from sklearn.cluster import kmeans_plusplus
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
X = np.load("glove100.npy").astype(np.float64)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
X = X[np.random.default_rng(0).permutation(len(X))]
Xtr, Xte, k, reg, seed = X[:320_000], X[320_000:], 256, 1e-6, 0
name = "GloVe-100 normalized (320k train / 80k test, k=256)"


def runs():
    for init, iters in [("kmeans", [1, 3, 10, 30]), ("random_from_data", [3, 10, 30, 100])]:
        for it in iters:

            def fit(it=it, init=init):
                g = GaussianMixture(
                    k, covariance_type="diag", init_params=init, max_iter=it, tol=0, reg_covar=reg, random_state=seed
                ).fit(Xtr)
                return g.weights_, g.means_, g.covariances_

            yield f"EM GaussianMixture (init={init})", it, fit
    for ep in [1, 2, 3, 5]:
        for init, label, B, a in [
            ("random rows", "law", None, None),
            ("random rows", "B=2048, a=0.2", 2048, 0.2),
            ("k-means++", "law", None, None),
        ]:

            def fit(ep=ep, init=init, B=B, a=a):
                if init == "k-means++":
                    M, _ = kmeans_plusplus(Xtr, k, random_state=seed)
                else:
                    M = Xtr[np.random.default_rng(seed).choice(len(Xtr), k, replace=False)]
                return egmm(Xtr, M, n_epochs=ep, batch_size=B, alpha=a, reg_covar=reg, seed=seed)

            yield f"e-GMM (init={init}, {label})", ep, fit


rows = []
for method, it, fit in runs():
    tic = time.perf_counter()
    p = fit()
    sec = time.perf_counter() - tic
    rows.append(
        dict(
            dataset=name,
            seed=seed,
            method=method,
            iters=it,
            seconds=sec,
            train_ll=egmm_score(Xtr, *p),
            test_ll=egmm_score(Xte, *p),
        )
    )
    print(method, it, round(sec, 1), round(rows[-1]["test_ll"], 3), flush=True)
    pl.DataFrame(rows).write_csv("results/frontier_glove.csv")
print("DONE", flush=True)
