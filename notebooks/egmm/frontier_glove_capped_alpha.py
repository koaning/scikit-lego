# ruff: noqa: T201, C408
import time
import warnings

import emeans as emeans_rust
import numpy as np
import polars as pl
from egmm import egmm, egmm_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
X = np.load("glove100.npy").astype(np.float64)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
X = X[np.random.default_rng(0).permutation(len(X))]
Xtr, Xte, k, reg, seed = X[:320_000], X[320_000:], 256, 1e-6, 0
X32 = np.ascontiguousarray(Xtr.astype(np.float32))
name = "GloVe-100 normalized (320k train / 80k test, k=256)"


def rows_init():
    return Xtr[np.random.default_rng(seed).choice(len(Xtr), k, replace=False)]


def emeans_init():
    return np.asarray(emeans_rust.Emeans(100, k, nepochs=5.0, seed=seed).train(X32)[0], dtype=np.float64)


def runs():
    for a in [0.1, 0.3]:
        for ep in [1, 2, 3, 5, 10]:
            yield (
                f"e-GMM (init=random rows, B=16384, a={a})",
                ep,
                lambda ep=ep, a=a: egmm(
                    Xtr, rows_init(), n_epochs=ep, batch_size=16384, alpha=a, reg_covar=reg, seed=seed
                ),
            )
    for ep in [1, 2, 3, 5]:
        yield (
            "e-GMM (init=e-means, B=16384, a=0.3)",
            ep,
            lambda ep=ep: egmm(Xtr, emeans_init(), n_epochs=ep, batch_size=16384, alpha=0.3, reg_covar=reg, seed=seed),
        )
    for it in [1, 3, 10, 30]:

        def fit(it=it):
            g = GaussianMixture(
                k,
                covariance_type="diag",
                init_params="random_from_data",
                means_init=emeans_init(),
                max_iter=it,
                tol=0,
                reg_covar=reg,
                random_state=seed,
            ).fit(Xtr)
            return g.weights_, g.means_, g.covariances_

        yield "EM GaussianMixture (init=e-means)", it, fit


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
    print(method, it, round(sec, 1), round(rows[-1]["test_ll"], 3), round(rows[-1]["train_ll"], 3), flush=True)
    pl.DataFrame(rows).write_csv("results/frontier_glove_capped_alpha.csv")
print("DONE", flush=True)
