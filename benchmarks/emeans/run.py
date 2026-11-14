# ruff: noqa: T201
"""Benchmark `EMeans` and `EGaussianMixture` against scikit-learn, over each method's full budget range.

Every method is fitted with a range of budgets (Lloyd/EM iterations or epochs), timed from start to finish
(initialisation included), and scored on held-out rows. scikit-learn's defaults, which iterate until they converge,
are the reference.

```
python benchmarks/emeans/run.py kmeans            # writes benchmarks/emeans/results/kmeans.csv
python benchmarks/emeans/run.py gmm               # writes benchmarks/emeans/results/gmm.csv
python benchmarks/emeans/run.py kmeans --quick    # small data, one seed: a smoke test in about a minute
python benchmarks/emeans/run.py kmeans --methods "EMeans (random)" --append   # rerun one method, keep the rest
```

The `backend="rust"` methods need `uv pip install ./rust` and are skipped without it. To add an implementation, add a `Method` to `KMEANS_METHODS` or `GMM_METHODS`. Its `make`
returns an unfitted estimator with `fit(X)`. k-means estimators must set `cluster_centers_`; mixtures must provide
`score`, `predict` and `sample`. Methods whose `available()` returns False are skipped.
"""

import argparse
import importlib.util
import platform
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_covtype, make_blobs
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin_min, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklego.cluster import EMeans
from sklego.mixture import EGaussianMixture

RESULTS = Path(__file__).parent / "results"
CONVERGED = "converged"


# ---------------------------------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray | None
    n_clusters: int  # used by the k-means benchmark
    n_components: int  # used by the mixture benchmark


def _split(name, X, y, n_clusters, n_components, quick):
    order = np.random.default_rng(0).permutation(len(X))
    if quick:
        order = order[:25_000]
        n_clusters, n_components = min(n_clusters, 64), min(n_components, 32)
    X, y = X[order], None if y is None else y[order]
    n_train = int(0.8 * len(X))
    return Dataset(name, X[:n_train], X[n_train:], None if y is None else y[n_train:], n_clusters, n_components)


def embedding_like(n_samples):
    """Many overlapping clusters on the unit sphere: structure like text embeddings, without a few clean clusters."""
    X, _ = make_blobs(n_samples=n_samples, n_features=100, centers=2_000, cluster_std=6.0, random_state=2)
    return X / np.linalg.norm(X, axis=1, keepdims=True), None


def covtype():
    """The 10 numeric columns of covtype, standardised (581k rows). Downloaded and cached by scikit-learn."""
    X = fetch_covtype().data[:, :10].astype(np.float64)
    return (X - X.mean(axis=0)) / X.std(axis=0), None


def blobs():
    """64 anisotropic Gaussian blobs of different widths, with true labels to measure cluster recovery."""
    rng = np.random.default_rng(1)
    X, y = make_blobs(
        n_samples=125_000, n_features=16, centers=64, cluster_std=rng.uniform(0.5, 2.5, 64), random_state=0
    )
    return X * rng.uniform(0.5, 2.0, 16), y


def load_datasets(task, names, quick):
    # k-means gets twice as many embedding rows as the mixtures, whose EM baselines are much slower.
    specs = {
        "embedding-like": lambda: (*embedding_like(250_000 if task == "kmeans" else 125_000), 256, 64),
        "covtype": lambda: (*covtype(), 256, 32),
        "blobs": lambda: (*blobs(), 64, 64),
    }
    for name in names:
        X, y, n_clusters, n_components = specs[name]()
        yield _split(name, X, y, n_clusters, n_components, quick)


# ---------------------------------------------------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    name: str  # unique, shown in tables and legends
    algorithm: str  # colour group in the charts
    init: str  # line style in the charts: "k-means++" is solid, anything else dashed
    unit: str  # "it" or "ep", shown next to budgets
    budgets: tuple  # iterations or epochs; (CONVERGED,) for a reference that stops by itself
    make: Callable  # k-means: make(n_clusters, budget, seed); mixtures: make(n_components, budget, seed, covariance)
    reference: bool = False
    covariances: tuple = ("diag", "spherical")  # mixtures only
    realism: tuple = ()  # mixtures only: budgets for which sample realism is measured (slow)
    max_seeds: int | None = None
    available: Callable = field(default=lambda: True)


class EmeansCrate:
    """Adapter for the third-party Rust `emeans` package (`pip install emeans`), included as a reference."""

    def __init__(self, n_clusters, n_epochs, seed):
        self.n_clusters, self.n_epochs, self.seed = n_clusters, n_epochs, seed

    def fit(self, X):
        import emeans

        X32 = np.ascontiguousarray(X, dtype=np.float32)
        centers, _ = emeans.Emeans(X.shape[1], self.n_clusters, nepochs=float(self.n_epochs), seed=self.seed).train(X32)
        self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        return self


def _installed(module):
    return lambda: importlib.util.find_spec(module) is not None


ITERATIONS = (1, 2, 5, 10, 20, 50, 100)
EPOCHS = (1, 2, 5, 10, 20, 50)

KMEANS_METHODS = [
    Method(
        "KMeans() defaults (runs until converged)",
        "KMeans",
        "k-means++",
        "it",
        (CONVERGED,),
        lambda k, b, s: KMeans(k, random_state=s),
        reference=True,
    ),
    Method("EMeans (random)", "EMeans", "random", "ep", EPOCHS, lambda k, b, s: EMeans(k, n_epochs=b, random_state=s)),
    Method(
        "EMeans (k-means++)",
        "EMeans",
        "k-means++",
        "ep",
        EPOCHS,
        lambda k, b, s: EMeans(k, n_epochs=b, init="k-means++", random_state=s),
    ),
    Method(
        "EMeans (Rust, random)",
        "EMeans (Rust)",
        "random",
        "ep",
        EPOCHS,
        lambda k, b, s: EMeans(k, n_epochs=b, random_state=s, backend="rust"),
        available=_installed("sklego_rust"),
    ),
    Method(
        "EMeans (Rust, k-means++)",
        "EMeans (Rust)",
        "k-means++",
        "ep",
        EPOCHS,
        lambda k, b, s: EMeans(k, n_epochs=b, init="k-means++", random_state=s, backend="rust"),
        available=_installed("sklego_rust"),
    ),
    Method(
        "KMeans (random), stopped after N iterations",
        "KMeans",
        "random",
        "it",
        ITERATIONS,
        lambda k, b, s: KMeans(k, init="random", n_init=1, max_iter=b, tol=0, random_state=s),
    ),
    Method(
        "KMeans (k-means++), stopped after N iterations",
        "KMeans",
        "k-means++",
        "it",
        ITERATIONS,
        lambda k, b, s: KMeans(k, n_init=1, max_iter=b, tol=0, random_state=s),
    ),
    Method(
        "MiniBatchKMeans, stopped after N epochs",
        "MiniBatchKMeans",
        "k-means++",
        "ep",
        (1, 2, 5, 10, 20),
        lambda k, b, s: MiniBatchKMeans(k, max_iter=b, max_no_improvement=None, tol=0, random_state=s),
    ),
    Method(
        "emeans crate (Rust, random)",
        "emeans crate (Rust)",
        "random",
        "ep",
        EPOCHS,
        lambda k, b, s: EmeansCrate(k, b, s),
        available=_installed("emeans"),
    ),
]

GMM_METHODS = [
    Method(
        "GaussianMixture() defaults (runs until converged)",
        "GaussianMixture",
        "k-means",
        "it",
        (CONVERGED,),
        lambda k, b, s, c: GaussianMixture(k, covariance_type=c, random_state=s),
        reference=True,
        realism=(CONVERGED,),
    ),
    Method(
        "EGaussianMixture",
        "EGaussianMixture",
        "e-means",
        "ep",
        EPOCHS,
        lambda k, b, s, c: EGaussianMixture(k, covariance_type=c, n_epochs=b, random_state=s),
        realism=(5, 50),
    ),
    Method(
        "EGaussianMixture (Rust)",
        "EGaussianMixture (Rust)",
        "e-means",
        "ep",
        EPOCHS,
        lambda k, b, s, c: EGaussianMixture(k, covariance_type=c, n_epochs=b, random_state=s, backend="rust"),
        realism=(5, 50),
        available=_installed("sklego_rust"),
    ),
    Method(
        "GaussianMixture, stopped after N EM iterations",
        "GaussianMixture",
        "k-means",
        "it",
        ITERATIONS,
        lambda k, b, s, c: GaussianMixture(k, covariance_type=c, max_iter=b, tol=0, random_state=s),
    ),
    # Not a competitor (EGaussianMixture has no full covariances): shows how much a full covariance would add.
    Method(
        "GaussianMixture() defaults, full covariance",
        "GaussianMixture (full)",
        "k-means",
        "it",
        (CONVERGED,),
        lambda k, b, s, c: GaussianMixture(k, covariance_type=c, random_state=s),
        covariances=("full",),
        realism=(CONVERGED,),
        max_seeds=1,
    ),
]


# ---------------------------------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------------------------------


def machine():
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        cpu = ""
    return cpu or platform.processor() or platform.machine()


def sample_realism(samples, real, seed):
    """AUC of a classifier telling `samples` from real held-out rows: 0.5 means indistinguishable."""
    Z, target = np.vstack([real, samples]), np.r_[np.zeros(len(real)), np.ones(len(samples))]
    # sample() returns rows grouped by component, so the folds must be shuffled.
    folds = StratifiedKFold(2, shuffle=True, random_state=seed)
    proba = cross_val_predict(
        HistGradientBoostingClassifier(random_state=seed), Z, target, cv=folds, method="predict_proba"
    )
    return float(roc_auc_score(target, proba[:, 1]))


def timed_fit(estimator, X):
    tic = time.perf_counter()
    estimator.fit(X)
    return estimator, time.perf_counter() - tic


def run_kmeans(data, method, budget, seed):
    model, seconds = timed_fit(method.make(data.n_clusters, budget, seed), data.X_train)
    labels, distances = pairwise_distances_argmin_min(data.X_test, model.cluster_centers_)
    yield dict(
        k=data.n_clusters,
        seconds=seconds,
        passes=float(getattr(model, "n_iter_", budget if budget != CONVERGED else np.nan)),
        test_mse=float(np.mean(distances**2)),
        ari=None if data.y_test is None else adjusted_rand_score(data.y_test, labels),
    )


def run_gmm(data, method, budget, seed, real):
    for covariance in method.covariances:
        model, seconds = timed_fit(method.make(data.n_components, budget, seed, covariance), data.X_train)
        yield dict(
            k=data.n_components,
            covariance=covariance,
            seconds=seconds,
            passes=float(getattr(model, "n_iter_", np.nan)),
            test_ll=float(model.score(data.X_test)),
            ari=None if data.y_test is None else adjusted_rand_score(data.y_test, model.predict(data.X_test)),
            realism=sample_realism(model.sample(len(real))[0], real, seed) if budget in method.realism else None,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task", choices=["kmeans", "gmm"])
    parser.add_argument("--data", nargs="+", default=["embedding-like", "covtype", "blobs"])
    parser.add_argument("--methods", nargs="+", default=None, help="method names to run (default: all available)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 0 1 2, or 0 with --quick")
    parser.add_argument("--quick", action="store_true", help="25k rows, fewer budgets and one seed")
    parser.add_argument("--append", action="store_true", help="replace only the rows of the methods that are rerun")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    out = args.out or RESULTS / f"{args.task}{'_quick' if args.quick else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or ([0] if args.quick else [0, 1, 2])
    methods = [m for m in (KMEANS_METHODS if args.task == "kmeans" else GMM_METHODS) if m.available()]
    if args.methods:
        unknown = set(args.methods) - {m.name for m in methods}
        if unknown:
            raise SystemExit(
                f"Unknown or unavailable methods: {sorted(unknown)}. Available: {[m.name for m in methods]}"
            )
        methods = [m for m in methods if m.name in args.methods]

    keep = pl.read_csv(out, infer_schema_length=None) if args.append and out.exists() else None
    if keep is not None:
        names = [m.name for m in methods] + ["real rows (floor)"]
        keep = keep.filter(~(pl.col("method").is_in(names) & pl.col("data").is_in(args.data)))
    rows, cpu = [], machine()

    def record(**row):
        rows.append({"task": args.task, "seed": 0, "machine": cpu, **row})
        frame = pl.DataFrame(rows, infer_schema_length=None)
        (frame if keep is None else pl.concat([keep, frame], how="diagonal_relaxed")).write_csv(out)

    for data in load_datasets(args.task, args.data, args.quick):
        real = data.X_test[
            np.random.default_rng(0).choice(len(data.X_test), min(20_000, len(data.X_test)), replace=False)
        ]
        for method in methods:
            budgets = method.budgets
            if args.quick and len(budgets) > 2:
                budgets = tuple(dict.fromkeys(budgets[::2] + budgets[-1:]))
            for budget in budgets:
                for seed in seeds[: method.max_seeds]:
                    if args.task == "kmeans":
                        results = run_kmeans(data, method, budget, seed)
                    else:
                        results = run_gmm(data, method, budget, seed, real)
                    for result in results:
                        record(
                            data=data.name,
                            method=method.name,
                            algorithm=method.algorithm,
                            init=method.init,
                            unit=method.unit,
                            reference=method.reference,
                            budget=str(budget),
                            seed=seed,
                            **result,
                        )
                        print(
                            f"{data.name:>15} | {method.name:<45} {result.get('covariance', ''):>9} {budget!s:>9} "
                            f"{method.unit} seed={seed} {result['seconds']:7.2f}s",
                            flush=True,
                        )
        if args.task == "gmm":
            # How well a classifier separates real training rows from real test rows: the best any model can do.
            floor = data.X_train[np.random.default_rng(1).choice(len(data.X_train), len(real), replace=False)]
            record(
                data=data.name,
                method="real rows (floor)",
                algorithm="floor",
                seconds=0.0,
                realism=sample_realism(floor, real, 0),
            )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
