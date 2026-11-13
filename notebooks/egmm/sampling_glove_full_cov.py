# ruff: noqa: T201, C408
import time
import warnings

import emeans as emeans_rust
import numpy as np
import polars as pl
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
    for it in [5, 20]:

        def fit(it=it):
            g = GaussianMixture(
                k,
                covariance_type="full",
                init_params="random_from_data",
                means_init=emeans_init(),
                max_iter=it,
                tol=0,
                reg_covar=reg,
                random_state=seed,
            ).fit(Xtr)
            print("converged", g.converged_, flush=True)
            return g

        yield "EM full covariance, e-means init", f"{it} it", fit


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
    if True:
        tic = time.perf_counter()
        g = fit()
        sec = time.perf_counter() - tic
        ll, S = float(g.score(Xte)), g.sample(2 * half)[0][rng.permutation(2 * half)]
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
    pl.DataFrame(rows).write_csv("results/sampling_glove_full_cov.csv")
print("DONE", flush=True)
