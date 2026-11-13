# ruff: noqa: T201, C408
import time
import warnings

import emeans as emeans_rust
import numpy as np
import polars as pl
from egmm import egmm, egmm_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
X = np.load("glove100.npy").astype(np.float64)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
X = X[np.random.default_rng(0).permutation(len(X))]
Xtr, Xte, k, reg, seed = X[:320_000], X[320_000:], 256, 1e-6, 0
X32 = np.ascontiguousarray(Xtr.astype(np.float32))


def emeans_init():
    return np.asarray(emeans_rust.Emeans(100, k, nepochs=5.0, seed=seed).train(X32)[0], dtype=np.float64)


def sk(**kw):
    g = GaussianMixture(k, covariance_type="diag", random_state=seed, **kw).fit(Xtr)
    return g.weights_, g.means_, g.covariances_, g.n_iter_


def models():
    yield "real train rows (floor)", None, None
    for ep, a in [(1, 0.3), (5, 0.3), (10, 0.3), (20, 0.3), (10, 0.1)]:
        yield (
            "e-GMM, e-means init",
            f"{ep} ep, alpha={a}",
            lambda ep=ep, a=a: egmm(
                Xtr, emeans_init(), n_epochs=ep, batch_size=16384, alpha=a, reg_covar=reg, seed=seed
            ),
        )
    yield (
        "EM, e-means init",
        "10 it",
        lambda: sk(init_params="random_from_data", means_init=emeans_init(), max_iter=10, tol=0)[:3],
    )

    def dflt():
        w, m, v, it = sk()
        print("default n_iter", it, flush=True)
        return w, m, v

    yield "GaussianMixture() defaults", "converged", dflt


rng = np.random.default_rng(1)
half = len(Xte) // 2
te_fit, te_eval = Xte[:half], Xte[half:]
nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
ref_nn = float(nn.kneighbors(te_eval[:5000])[0].mean())


def sample(w, m, v, n):
    comp = rng.choice(len(w), size=n, p=w / w.sum())
    return m[comp] + np.sqrt(v[comp]) * rng.standard_normal((n, m.shape[1]))


def c2st(S):
    S_fit, S_eval = S[:half], S[half : 2 * half]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(
        np.vstack([te_fit, S_fit]), np.r_[np.zeros(half), np.ones(half)]
    )
    return roc_auc_score(np.r_[np.zeros(half), np.ones(half)], clf.predict_proba(np.vstack([te_eval, S_eval]))[:, 1])


rows = []
for method, label, fit in models():
    if fit is None:
        sec, ll, S = 0.0, float("nan"), Xtr[rng.choice(len(Xtr), 2 * half, replace=False)]
    else:
        tic = time.perf_counter()
        p = fit()
        sec = time.perf_counter() - tic
        ll, S = egmm_score(Xte, *p), sample(*p, 2 * half)
    Sn = S / np.linalg.norm(S, axis=1, keepdims=True)
    row = dict(
        method=method,
        label=label,
        seconds=sec,
        test_ll=ll,
        auc_raw=c2st(S),
        auc_renorm=c2st(Sn),
        nn_sample_to_train=float(nn.kneighbors(Sn[:5000])[0].mean()),
        nn_test_to_train=ref_nn,
        sample_norm_mean=float(np.linalg.norm(S, axis=1).mean()),
    )
    rows.append(row)
    print(row, flush=True)
    pl.DataFrame(rows).write_csv("results/sampling_glove.csv")
print("DONE", flush=True)
