# Clustering

## e-means

[`EMeans`][emeans-api] is a drop-in alternative to `KMeans` that usually reaches the same (or a lower) error in a
fraction of the time. It is a NumPy port of [e-means](https://arxiv.org/abs/2611.09501), from the
[faiss](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) team.

Instead of recomputing every centroid from all of the data in each iteration, e-means walks over the data in
mini-batches. Every batch is assigned to its nearest centroids, after which each centroid $c_j$ with mass $N_j$ moves
towards the mean of the $n_j$ rows it received (summing to $s_j$):

$$
N_j \leftarrow (1 - \alpha_t) N_j + \alpha_t n_j, \qquad
c_j \leftarrow c_j + \frac{\alpha_t n_j}{N_j} \left(\frac{s_j}{n_j} - c_j\right)
$$

The step size $\alpha_t$ follows a cosine decay from `alpha` down to `alpha / 10`. Clusters that receive almost no
rows are revived by splitting a busy one, and after the last pass every centroid is replaced by the exact mean of its
rows.

```py
from sklearn.datasets import make_blobs
from sklego.cluster import EMeans

X, _ = make_blobs(n_samples=100_000, centers=50, n_features=32, random_state=42)

emeans = EMeans(n_clusters=50, random_state=42).fit(X)
emeans.predict(X[:5])
emeans.batch_size_, emeans.alpha_
```

There are two knobs: `batch_size` and `alpha`. By default both are `None`, so the scaling laws from the paper pick
them based on the number of rows, clusters and epochs. You usually don't need to tune anything beyond `n_clusters` and
`n_epochs` (more epochs is slower but better, the default is 5).

!!! tip "When to use `init=\"k-means++\"`"

    The default starts from random rows, which works well on embeddings and other data without a few, clean clusters.
    On data with well separated clusters, or tabular data with rare outliers, random starts can put two centroids in
    the same cluster. For data like that, use `init="k-means++"` and consider `unused_threshold=0` to turn revival
    off.

### How does it compare?

[`benchmarks/emeans`](https://github.com/koaning/scikit-lego/tree/main/benchmarks/emeans) fits `EMeans`, `KMeans` and
`MiniBatchKMeans` over their full range of iterations or epochs, and scores every fit on held-out rows. In short:

- On embedding-like data (many overlapping clusters, 100 features), `EMeans` reaches the test error of `KMeans()` with
  its defaults several times faster, and with a few more epochs ends slightly *below* it.
- On covtype, `EMeans` gets within half a percent of `KMeans()` many times faster, but fully converged `KMeans()` still
  ends a few tenths of a percent lower.
- On data with well separated clusters, random starts fail for `EMeans` just as they do for `KMeans(init="random")`.
  Use `init="k-means++"` there.
- `MiniBatchKMeans` ends 1.5–4.5% behind on all three.

The [benchmark README](https://github.com/koaning/scikit-lego/tree/main/benchmarks/emeans) has the charts, tables and
the commands to rerun it.

### Rust backend

`EMeans(backend="rust")` runs the training loop and the full-data passes (`labels_`, `predict`, `transform`, `score`)
in compiled Rust. Every other parameter means the same thing, and the defaults are the same. The extension is built
from the `rust/` folder of the repository and needs a [Rust toolchain](https://rustup.rs):

```bash
pip install ./rust
```

```py
emeans = EMeans(n_clusters=50, random_state=42, backend="rust").fit(X)
```

Both backends run the same algorithm, but they draw different random numbers, so a given `random_state` gives a
different (equally good) model on each. In the benchmark (Apple M4 Pro, mean over 3 seeds, `n_clusters=256`, same
number of epochs on both backends):

| data | epochs | `backend="numpy"` | `backend="rust"` | speedup | test MSE, numpy | test MSE, rust |
|---|--:|--:|--:|--:|--:|--:|
| embedding-like: 200k rows, 100 features | 5 | 0.68s | 0.24s | 2.8x | 0.8593 | 0.8595 |
| | 50 | 6.61s | 1.68s | 3.9x | 0.8581 | 0.8582 |
| covtype: 465k rows, 10 features | 5 | 0.90s | 0.24s | 3.7x | 1.4180 | 1.4169 |
| | 50 | 8.30s | 1.30s | 6.4x | 1.4149 | 1.4114 |

The gain is largest with few features, where NumPy spends most of its time on per-batch overhead. With many features,
both backends spend most of their time in the same BLAS matrix products. With `init="k-means++"` the speedup is
smaller (1.1–2.6x), because both backends use scikit-learn's k-means++, which takes about 2 seconds on these datasets.

The same benchmark covers [`EGaussianMixture`][egmm-api], which applies the e-means update to Gaussian mixtures. See
the [mixture methods](mixture-methods.md#fast-gaussian-mixtures-with-egaussianmixture) page.

[emeans-api]: ../../api/cluster#sklego.cluster.emeans.EMeans
[egmm-api]: ../../api/mixture#sklego.mixture.egmm.EGaussianMixture
