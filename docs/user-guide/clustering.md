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

### How much faster?

`notebooks/emeans_benchmark.py` is a [marimo](https://marimo.io) notebook that times `EMeans` against `KMeans` and
`MiniBatchKMeans`, scoring every fit on held-out rows (covtype is one of the datasets in its dropdown). On one run (Apple M4 Pro,
one seed, `n_clusters=256`, time includes initialisation) it found:

| data | `KMeans()` defaults | `EMeans()` defaults | `EMeans(init="k-means++", unused_threshold=0)` |
|---|---|---|---|
| embedding-like: 100k rows, 100-d, unit norm | 3.5s, test MSE 0.8640 | **0.43s**, 0.8645 (10 epochs: 0.72s, 0.8637) | - |
| covtype: 464k rows, 10 numeric columns | 11.5s, test MSE 1.406 | **0.90s**, 1.421 | 4.0s, 1.406 |

On embedding-like data the defaults match `KMeans` about 8 times faster. On covtype, random starts give up about 1%
of the error, and starting from k-means++ closes that gap while staying almost 3 times faster.

The same notebook also benchmarks [`EGaussianMixture`][egmm-api], which applies the e-means update to Gaussian
mixtures. See the [mixture methods](mixture-methods.md#fast-gaussian-mixtures-with-egaussianmixture) page.

[emeans-api]: ../../api/cluster#sklego.cluster.emeans.EMeans
[egmm-api]: ../../api/mixture#sklego.mixture.egmm.EGaussianMixture
