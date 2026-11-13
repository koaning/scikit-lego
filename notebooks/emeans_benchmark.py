import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    import warnings

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.datasets import fetch_covtype, make_blobs
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.mixture import GaussianMixture

    from sklego.cluster import EMeans
    from sklego.mixture import EGaussianMixture

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    return (
        EGaussianMixture,
        EMeans,
        GaussianMixture,
        KMeans,
        MiniBatchKMeans,
        alt,
        fetch_covtype,
        make_blobs,
        mo,
        np,
        pairwise_distances_argmin_min,
        pl,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How fast are `EMeans` and `EGaussianMixture`?

    [e-means](https://arxiv.org/abs/2611.09501) trains k-means with mini-batches and an *exponential moving average*
    of the centroids. For every batch, each centroid $c_j$ with mass $N_j$ that receives $n_j$ rows summing to $s_j$
    moves as

    $$
    N_j \leftarrow (1 - \alpha_t) N_j + \alpha_t n_j, \qquad
    c_j \leftarrow c_j + \frac{\alpha_t n_j}{N_j} \left(\frac{s_j}{n_j} - c_j\right)
    $$

    with $\alpha_t$ decaying from $\alpha$ to $\alpha / 10$. `EGaussianMixture` applies the same update to the
    soft sufficient statistics of a Gaussian mixture ($\sum_i r_{ij}$, $\sum_i r_{ij} x_i$, $\sum_i r_{ij} x_i^2$),
    so every mini-batch is an E-step followed by a moving average instead of a full M-step.

    Both estimators in scikit-lego are plain NumPy. This notebook checks how much time they save against
    scikit-learn's `KMeans`, `MiniBatchKMeans` and `GaussianMixture`. Every fit is timed from start to finish
    (initialisation included) and scored on held-out rows. Run it from the root of the repo with:

    ```
    uv pip install -e . marimo altair polars
    marimo edit notebooks/emeans_benchmark.py   # or: python notebooks/emeans_benchmark.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dataset = mo.ui.dropdown(
        ["embedding-like (100-d, unit norm)", "covtype (10 numeric columns)", "blobs (16-d)"],
        value="embedding-like (100-d, unit norm)",
        label="dataset",
    )
    n_samples = mo.ui.slider(20_000, 400_000, step=20_000, value=100_000, label="training rows", show_value=True)
    mo.vstack([mo.md("## Data"), mo.hstack([dataset, n_samples], justify="start")])
    return dataset, n_samples


@app.cell
def _(dataset, fetch_covtype, make_blobs, n_samples, np):
    # 25% extra rows are kept apart as a held-out test set.
    _n = int(n_samples.value * 1.25)
    if dataset.value.startswith("embedding"):
        # Many overlapping clusters projected on the unit sphere, which behaves a lot like text embeddings:
        # there is structure, but no small number of clean clusters for Lloyd's algorithm to snap into.
        _X, _ = make_blobs(n_samples=_n, n_features=100, centers=2_000, cluster_std=6.0, random_state=2)
        _X /= np.linalg.norm(_X, axis=1, keepdims=True)
    elif dataset.value.startswith("covtype"):
        _X = fetch_covtype().data[:, :10].astype(np.float64)
        _X = (_X - _X.mean(axis=0)) / _X.std(axis=0)
        _X = _X[np.random.default_rng(0).permutation(len(_X))[:_n]]
    else:
        _rng = np.random.default_rng(1)
        _X, _ = make_blobs(
            n_samples=_n, n_features=16, centers=64, cluster_std=_rng.uniform(0.5, 2.5, 64), random_state=0
        )
        _X = _X * _rng.uniform(0.5, 2.0, 16)
    _X = _X[np.random.default_rng(0).permutation(len(_X))]
    _split = int(len(_X) * 0.8)
    X, X_test = _X[:_split], _X[_split:]
    data_summary = f"{X.shape[0]:,} training rows, {X_test.shape[0]:,} test rows, {X.shape[1]} features"
    return X, X_test, data_summary


@app.cell
def _(alt, pl, time):
    def frontier_chart(df, y, y_title, ref, ref_label):
        """Quality against training time, one line per method, with a dashed line for the sklearn default."""
        _base = alt.Chart(df).encode(
            x=alt.X("seconds:Q", scale=alt.Scale(type="log"), title="training time incl. init (s, log scale)"),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False), title=y_title),
            color=alt.Color("method:N", title=None, legend=alt.Legend(orient="bottom", columns=2, labelLimit=300)),
            tooltip=["method", "label", alt.Tooltip("seconds:Q", format=".2f"), alt.Tooltip(f"{y}:Q", format=".4f")],
        )
        _rule = alt.Chart(pl.DataFrame({y: [ref], "text": [ref_label]})).encode(y=f"{y}:Q")
        return (
            _rule.mark_rule(strokeDash=[5, 4], color="gray")
            + _rule.mark_text(align="left", x=5, dy=-6, color="gray").encode(text="text:N")
            + _base.mark_line(point=True)
            + _base.mark_text(align="left", dx=6, dy=-6, fontSize=10).encode(text="label:N")
        ).properties(width=620, height=360)

    def speedup_table(df, y, ref_method, higher_is_better, tolerance):
        """Per method: the fastest fit that is at least as good as the reference (within `tolerance`)."""
        _ref = df.filter(pl.col("method") == ref_method).row(0, named=True)
        _good = pl.col(y) >= _ref[y] - tolerance if higher_is_better else pl.col(y) <= _ref[y] * (1 + tolerance)
        _best = pl.col(y).max() if higher_is_better else pl.col(y).min()
        _fastest = df.filter(_good).sort("seconds").group_by("method", maintain_order=True).first()
        return (
            df.group_by("method")
            .agg(best=_best)
            .join(_fastest.select("method", "label", "seconds"), on="method", how="left")
            .with_columns(speedup=_ref["seconds"] / pl.col("seconds"))
            .rename({"label": "fastest fit that matches", "seconds": "its time (s)", "best": f"best {y}"})
            .sort("speedup", descending=True, nulls_last=True)
        )

    def timed(fit):
        """Run `fit()` and return its result and the wall-clock time in seconds."""
        _tic = time.perf_counter()
        _model = fit()
        return _model, time.perf_counter() - _tic

    return frontier_chart, speedup_table, timed


@app.cell(hide_code=True)
def _(mo, data_summary):
    kmeans_k = mo.ui.slider(32, 1024, step=32, value=256, label="n_clusters", show_value=True)
    mo.vstack(
        [
            mo.md(rf"""
    ## k-means: `EMeans` against `KMeans` and `MiniBatchKMeans`

    {data_summary}. Lines connect fits of the same method with more iterations (`it`) or epochs (`ep`). The y-axis is
    the mean squared distance of held-out rows to their nearest centroid (lower is better). The dashed line is
    `KMeans(n_clusters).fit(X)` with all of scikit-learn's defaults, which is what most people run. Fits more than
    3% worse than that line are left out of the chart.
    """),
            kmeans_k,
        ]
    )
    return (kmeans_k,)


@app.cell
def _(EMeans, KMeans, MiniBatchKMeans, X, X_test, kmeans_k, np, pairwise_distances_argmin_min, pl, timed):
    def _mse(data, centers):
        return float(np.mean(pairwise_distances_argmin_min(data, centers)[1] ** 2))

    def _kmeans_fits(k):
        yield "sklearn KMeans() defaults", "converged", lambda: KMeans(k, random_state=0).fit(X)
        for _it in [3, 10, 30]:
            yield (
                "KMeans (k-means++)",
                f"{_it} it",
                lambda _it=_it: KMeans(k, n_init=1, max_iter=_it, tol=0, random_state=0).fit(X),
            )
        for _it in [3, 10, 30, 100]:
            yield (
                "KMeans (random)",
                f"{_it} it",
                lambda _it=_it: KMeans(k, init="random", n_init=1, max_iter=_it, tol=0, random_state=0).fit(X),
            )
        for _it in [1, 3, 10]:
            yield (
                "MiniBatchKMeans (B=4096)",
                f"{_it} it",
                lambda _it=_it: MiniBatchKMeans(
                    k, batch_size=4096, max_iter=_it, max_no_improvement=None, tol=0, random_state=0
                ).fit(X),
            )
        for _ep in [1, 2, 3, 5, 10]:
            yield "sklego EMeans", f"{_ep} ep", lambda _ep=_ep: EMeans(k, n_epochs=_ep, random_state=0).fit(X)

    _rows = []
    for _method, _label, _fit in _kmeans_fits(kmeans_k.value):
        _model, _seconds = timed(_fit)
        _rows.append(
            {
                "method": _method,
                "label": _label,
                "seconds": _seconds,
                "test MSE": _mse(X_test, _model.cluster_centers_),
            }
        )
    kmeans_results = pl.DataFrame(_rows)
    return (kmeans_results,)


@app.cell
def _(frontier_chart, kmeans_results, mo, pl, speedup_table):
    _ref = kmeans_results.filter(pl.col("method") == "sklearn KMeans() defaults").row(0, named=True)
    mo.vstack(
        [
            frontier_chart(
                kmeans_results.filter(pl.col("test MSE") <= 1.03 * _ref["test MSE"]),
                "test MSE",
                "test MSE (lower is better)",
                _ref["test MSE"],
                f"sklearn KMeans() defaults: {_ref['seconds']:.1f}s",
            ),
            mo.md(
                "**Speedup**: time of `KMeans()` with defaults divided by the time of the fastest fit of each method "
                "that gets a test MSE within 0.1% of it. Empty means no fit got there."
            ),
            speedup_table(kmeans_results, "test MSE", "sklearn KMeans() defaults", False, 0.001),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, data_summary):
    gmm_k = mo.ui.slider(8, 256, step=8, value=64, label="n_components", show_value=True)
    mo.vstack(
        [
            mo.md(rf"""
    ## Gaussian mixtures: `EGaussianMixture` against `GaussianMixture`

    {data_summary}. Same setup with diagonal covariances; the y-axis is now the average log-likelihood of held-out rows
    (higher is better). The dashed line is `GaussianMixture(n_components, covariance_type="diag").fit(X)` with
    scikit-learn's defaults (k-means initialisation, EM until the lower bound improves by less than `1e-3`). Fits
    more than 3 nats/row below that line are left out of the chart.
    """),
            gmm_k,
        ]
    )
    return (gmm_k,)


@app.cell
def _(EGaussianMixture, GaussianMixture, X, X_test, gmm_k, pl, timed):
    def _gmm_fits(k, covariance_type):
        yield (
            f"sklearn GaussianMixture() defaults ({covariance_type})",
            "converged",
            lambda: GaussianMixture(k, covariance_type=covariance_type, random_state=0).fit(X),
        )
        if covariance_type == "spherical":
            for _ep in [2, 5]:
                yield (
                    "sklego EGaussianMixture (spherical)",
                    f"{_ep} ep",
                    lambda _ep=_ep: EGaussianMixture(k, covariance_type="spherical", n_epochs=_ep, random_state=0).fit(
                        X
                    ),
                )
            return
        for _it in [1, 3, 10, 30]:
            yield (
                "GaussianMixture (diag, k-means init)",
                f"{_it} it",
                lambda _it=_it: GaussianMixture(k, covariance_type="diag", max_iter=_it, tol=0, random_state=0).fit(X),
            )
        for _ep in [1, 2, 3, 5, 10]:
            yield (
                "sklego EGaussianMixture (diag)",
                f"{_ep} ep",
                lambda _ep=_ep: EGaussianMixture(k, n_epochs=_ep, random_state=0).fit(X),
            )

    _rows = []
    for _covariance_type in ["diag", "spherical"]:
        for _method, _label, _fit in _gmm_fits(gmm_k.value, _covariance_type):
            _model, _seconds = timed(_fit)
            _rows.append(
                {
                    "covariance_type": _covariance_type,
                    "method": _method,
                    "label": _label,
                    "seconds": _seconds,
                    "test log-likelihood": _model.score(X_test),
                }
            )
    gmm_results = pl.DataFrame(_rows)
    return (gmm_results,)


@app.cell
def _(frontier_chart, gmm_results, mo, pl, speedup_table):
    _diag = gmm_results.filter(pl.col("covariance_type") == "diag")
    _spherical = gmm_results.filter(pl.col("covariance_type") == "spherical")
    _ref = _diag.filter(pl.col("method").str.starts_with("sklearn")).row(0, named=True)
    mo.vstack(
        [
            frontier_chart(
                _diag.filter(pl.col("test log-likelihood") >= _ref["test log-likelihood"] - 3),
                "test log-likelihood",
                "held-out log-likelihood per row (higher is better)",
                _ref["test log-likelihood"],
                f"sklearn GaussianMixture() defaults: {_ref['seconds']:.1f}s",
            ),
            mo.md(
                "**Speedup** (diagonal): time of `GaussianMixture()` with defaults divided by the time of the fastest "
                "fit of each method whose test log-likelihood is at most 0.05 nats/row below it."
            ),
            speedup_table(_diag, "test log-likelihood", _ref["method"], True, 0.05),
            mo.md("**Spherical covariances**, same comparison:"),
            speedup_table(
                _spherical, "test log-likelihood", "sklearn GaussianMixture() defaults (spherical)", True, 0.05
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes

    - Both estimators default to the e-means scaling laws for the batch size and step size, so the numbers above
      need no tuning. For mixtures the step size is capped at 0.3 and the batch holds at least 64 rows per
      component: the k-means laws push $\alpha$ to 1 on longer runs, which turns the moving average into noisy
      minibatch EM (see `notebooks/egmm/README.md`).
    - `EGaussianMixture` starts from `EMeans` centroids seeded with k-means++ on a small subsample. scikit-learn's
      default initialisation runs a full `KMeans` first, which is often most of its time.
    - Larger offline runs on GloVe-100 (320k rows) are summarised in `notebooks/egmm/README.md`, including the
      Rust `emeans` crate as a reference for how much is left on the table by staying in NumPy.
    """)
    return


if __name__ == "__main__":
    app.run()
