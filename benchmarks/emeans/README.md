# EMeans and EGaussianMixture benchmark

Compares `sklego.cluster.EMeans` with `KMeans` and `MiniBatchKMeans`, and `sklego.mixture.EGaussianMixture` with
`GaussianMixture`. Every method is fitted over its full budget range (1–100 Lloyd or EM iterations, 1–50 epochs),
timed from start to finish (initialisation included), and scored on held-out rows. The reference is scikit-learn's
estimator with its defaults, which iterates until it converges.

The latest results are in [`results/tables.md`](results/tables.md), with the charts `results/kmeans.png` and
`results/gmm.png`.

## Running it

From the root of the repo, with the dev install of scikit-lego:

```bash
uv pip install -e . polars altair vl-convert-python
uv pip install ./rust                                  # optional: the backend="rust" methods (needs a Rust toolchain)
uv pip install emeans                                  # optional: the third-party Rust e-means, as a reference

python benchmarks/emeans/run.py kmeans                 # ~30 min on an M4 Pro
python benchmarks/emeans/run.py gmm                    # ~40 min
python benchmarks/emeans/report.py                     # charts + results/tables.md
```

`--quick` runs on 25k rows with fewer budgets and one seed, in a minute or two. It writes `*_quick.*` files, which
are not committed:

```bash
python benchmarks/emeans/run.py kmeans --quick && python benchmarks/emeans/run.py gmm --quick
python benchmarks/emeans/report.py --quick
```

Other options: `--data covtype blobs` limits the datasets, `--seeds 0 1` sets the seeds, and `--methods NAME ...` runs
only some methods. With `--append`, only the rows of the methods (and datasets) that are rerun are replaced, so a new
implementation can be added to existing results without rerunning everything.

## Methods

Every e-means and e-GMM method comes in two flavours: `backend="numpy"` (the default) and `backend="rust"`
(`EMeans (Rust, ...)`, `EGaussianMixture (Rust)`). Both run the same algorithm with the same defaults, so at the same
budget they give the same quality (up to which random numbers they draw) and the difference in time is the speedup of
the compiled loop. The Rust methods are skipped when `sklego_rust` is not installed.

## Adding an implementation

Add a `Method` to `KMEANS_METHODS` or `GMM_METHODS` in `run.py`:

```python
Method(
    "EMeans (Rust)",              # name in tables and legends
    "EMeans (Rust)",              # algorithm: one colour per algorithm in the charts
    "random",                     # init: "k-means++" is drawn solid, anything else dashed
    "ep",                         # unit of the budget
    EPOCHS,                       # budgets to sweep
    lambda k, b, s: RustEMeans(k, n_epochs=b, random_state=s),
    available=_installed("my_rust_module"),
),
```

`make` returns an unfitted estimator with `fit(X)`. k-means estimators must set `cluster_centers_` (and may set
`n_iter_`). Mixtures get `make(n_components, budget, seed, covariance)` and must provide `score`, `predict` and
`sample`. `EmeansCrate` in `run.py` is an example adapter, for the third-party
[`emeans`](https://pypi.org/project/emeans/) package. Then:

```bash
python benchmarks/emeans/run.py kmeans --methods "EMeans (Rust)" --append
python benchmarks/emeans/report.py
```

## Datasets

| name | rows (train / test) | features | k-means `k` | mixture components | why |
|---|---|---|---|---|---|
| embedding-like | 200k / 50k (k-means), 100k / 25k (mixtures) | 100 | 256 | 64 | 2,000 overlapping blobs on the unit sphere: structure like text embeddings, without a few clean clusters |
| covtype | 465k / 116k | 10 | 256 | 32 | real data: the numeric columns of covtype, standardised. Downloaded by `sklearn.datasets.fetch_covtype` |
| blobs | 100k / 25k | 16 | 64 | 64 | 64 anisotropic blobs with true labels, to measure cluster recovery (ARI) |

## What is measured

- **Quality against time** (the charts): one point per budget, as the mean over 3 seeds.
- **Time to reach the reference**: the first budget whose mean held-out score is within a tolerance of
  scikit-learn's default: 0.5% of the test MSE for k-means, 0.02 nats/row for mixtures. Change it with
  `report.py --kmeans-tolerance` and `--gmm-tolerance`.
- **Best score at any budget**: shows whether a method is only fast because it stops early.
- **Cluster recovery** on blobs: adjusted Rand index of each method's best fit against the true labels.
- **Sample realism** (mixtures): a `HistGradientBoostingClassifier` tries to tell 20k samples from 20k real held-out
  rows, using 2-fold cross-validation. 0.5 means indistinguishable. Two real samples set the floor.
- **Full covariance** (mixtures, one seed): what `GaussianMixture(covariance_type="full")` adds over the diagonal
  default. `EGaussianMixture` only supports `diag` and `spherical`.

## How the `EGaussianMixture` defaults were picked

These notes come from the research that led to `EGaussianMixture`, which included runs on GloVe-100 (320k rows) that
are not part of this benchmark.

- The k-means scaling laws set `alpha` to 1 for runs of 3+ epochs. For mixtures that turns the moving average into
  noisy mini-batch EM that gets *worse* with more epochs: on GloVe, from random rows, test log-likelihood ended 0.9–1.5
  nats/row behind EM. In sweeps over `batch_size` × `alpha` × epochs on four datasets, `alpha=1` lost 1.6–13.6
  nats/row whenever a batch held 64 or fewer rows per component.
- The stable region is roughly `batch_size / n_components >= 200 * alpha`. Both (64 rows per component, `alpha=0.3`)
  and (128, `alpha=0.6`) were stable everywhere, and matched each other to within 0.02 nats/row. (64, `alpha=0.6`)
  was 0.04–0.23 nats/row better on some data, but degraded with more epochs on another (88.96 to 88.55 nats/row
  from 3 to 10 epochs). So the defaults are `batch_size >= 64 * n_components` and `alpha <= 0.3`.
- Random e-means starts lost about 1.6 nats/row on well-separated blobs (two centroids in one blob).
  k-means++ on a subsample of `64 * n_components` rows matched k-means++ on all rows over 5 seeds, at 0.13s
  instead of 3.1s (320k × 100, 256 components). That is what `init="e-means"` uses.
- On GloVe, with e-means init, B=16384 and `alpha=0.3`, the diagonal prototype reached 109.72 nats/row in 31s.
  `GaussianMixture()` needed 106s for the same (almost all of it k-means initialisation), and converged at 109.95 after
  387s.
