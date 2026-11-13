import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.special import logsumexp
    from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
    from sklearn.datasets import make_blobs
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.mixture import GaussianMixture

    from sklego.cluster import EMeans
    from sklego.cluster.emeans import _epoch_rows as epoch_rows
    from sklego.cluster.emeans import _scheduled_alpha as scheduled_alpha
    from sklego.cluster.emeans import emeans_alpha, emeans_batch_size

    return (
        EMeans,
        GaussianMixture,
        KMeans,
        MiniBatchKMeans,
        alt,
        emeans_alpha,
        emeans_batch_size,
        epoch_rows,
        scheduled_alpha,
        kmeans_plusplus,
        logsumexp,
        make_blobs,
        math,
        mo,
        np,
        pairwise_distances_argmin_min,
        pl,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # e-means in scikit-lego

    [e-means](https://arxiv.org/abs/2611.09501) trains k-means with mini-batches and an *exponential moving average*
    of the centroids. For every batch, each centroid $c_j$ with mass $N_j$ that receives $n_j$ rows summing to $s_j$ is updated as

    $$
    N_j \leftarrow (1 - \alpha_t) N_j + \alpha_t n_j, \qquad
    c_j \leftarrow c_j + \frac{\alpha_t n_j}{N_j} \left(\frac{s_j}{n_j} - c_j\right)
    $$

    where $\alpha_t$ follows a cosine decay from $\alpha$ to $\alpha / 10$. Two knobs matter: the **batch size** $B$ and
    the starting **step size** $\alpha$. The paper fits scaling laws for both, so by default `EMeans` needs no tuning.
    This notebook checks how `sklego.cluster.EMeans` compares to Lloyd's k-means and `MiniBatchKMeans`, sweeps
    $\alpha$ and $B$ to see how close the scaling laws land to the best setting, and then tries the same trick on
    Gaussian mixture models.

    Run it from the root of the repo with:

    ```
    uv pip install -e . marimo altair polars emeans
    marimo edit notebooks/emeans_benchmark.py
    ```

    `emeans` (the [Rust crate](https://crates.io/crates/emeans) with Python bindings) is optional; when installed it is
    added to the head-to-head benchmark as a reference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dataset = mo.ui.dropdown(["gaussian noise", "blobs"], value="gaussian noise", label="dataset")
    n_samples = mo.ui.slider(5_000, 200_000, step=5_000, value=50_000, label="n_samples", show_value=True)
    n_features = mo.ui.slider(2, 128, step=2, value=32, label="n_features", show_value=True)
    n_clusters = mo.ui.slider(8, 512, step=8, value=128, label="n_clusters", show_value=True)
    mo.vstack([mo.md("## Data"), mo.hstack([dataset, n_samples, n_features, n_clusters], justify="start")])
    return dataset, n_clusters, n_features, n_samples


@app.cell
def _(dataset, make_blobs, n_clusters, n_features, n_samples, np):
    if dataset.value == "blobs":
        X, _ = make_blobs(
            n_samples=n_samples.value,
            n_features=n_features.value,
            centers=n_clusters.value,
            cluster_std=np.random.default_rng(0).uniform(0.5, 3.0, n_clusters.value),
            random_state=0,
        )
    else:
        # Random Gaussian data has no "true" clusters, which makes it a hard, honest test of the optimizer.
        X = np.random.default_rng(0).normal(size=(n_samples.value, n_features.value))
    X = X.astype(np.float32)
    k = n_clusters.value
    return X, k


@app.cell
def _(np, pairwise_distances_argmin_min):
    def inertia(X, centers):
        _, dist = pairwise_distances_argmin_min(X, centers)
        return float(np.square(dist, dtype=np.float64).sum())

    return (inertia,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Head to head

    All methods start from random rows (`n_init=1`) except where noted. *relative inertia* divides by Lloyd's k-means
    after 25 iterations, so values below 1 beat it. *epochs* counts passes over the data used for training.
    """)
    return


@app.cell
def _(EMeans, KMeans, MiniBatchKMeans, X, inertia, k, pl, time):
    def _timed(fit):
        tic = time.perf_counter()
        centers, epochs = fit()
        return time.perf_counter() - tic, centers, epochs

    def _lloyd(init):
        model = KMeans(k, init=init, n_init=1, max_iter=25, tol=0, random_state=0).fit(X)
        return model.cluster_centers_, model.n_iter_

    def _minibatch():
        model = MiniBatchKMeans(k, init="random", n_init=1, batch_size=1024, max_iter=25, random_state=0).fit(X)
        return model.cluster_centers_, model.n_steps_ * 1024 / len(X)

    def _emeans(n_epochs):
        def fit():
            model = EMeans(k, n_epochs=n_epochs, random_state=0).fit(X)
            return model.cluster_centers_, n_epochs

        return fit

    candidates = {
        "Lloyd k-means (random)": lambda: _lloyd("random"),
        "Lloyd k-means (k-means++)": lambda: _lloyd("k-means++"),
        "MiniBatchKMeans (B=1024)": _minibatch,
        "sklego EMeans (2 epochs)": _emeans(2),
        "sklego EMeans (5 epochs)": _emeans(5),
    }
    try:
        import emeans as emeans_rust

        def _rust():
            centers, _ = emeans_rust.Emeans(X.shape[1], k, nepochs=5.0, seed=0).train(X)
            return centers, 5

        candidates["emeans Rust crate (5 epochs)"] = _rust
    except ImportError:
        pass

    _rows = []
    for _name, _fit in candidates.items():
        _seconds, _centers, _epochs = _timed(_fit)
        _rows.append({"method": _name, "seconds": _seconds, "epochs": _epochs, "inertia": inertia(X, _centers)})
    head_to_head = pl.DataFrame(_rows).with_columns(
        relative_inertia=pl.col("inertia") / pl.col("inertia").filter(pl.col("method") == "Lloyd k-means (random)")
    )
    head_to_head
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Error versus passes over the data

    Lloyd's algorithm needs one pass per iteration. e-means can stop at any (fractional) number of passes.
    """)
    return


@app.cell
def _(EMeans, KMeans, X, alt, inertia, k, pl):
    _rows = []
    for _iters in [1, 2, 3, 5, 8, 12, 25]:
        _model = KMeans(k, init="random", n_init=1, max_iter=_iters, tol=0, random_state=0).fit(X)
        _rows.append({"method": "Lloyd k-means", "epochs": _iters, "inertia": inertia(X, _model.cluster_centers_)})
    for _epochs in [0.5, 1, 1.5, 2, 3, 5, 8]:
        _model = EMeans(k, n_epochs=_epochs, random_state=0).fit(X)
        _rows.append({"method": "EMeans", "epochs": _epochs, "inertia": _model.inertia_})
    epochs_df = pl.DataFrame(_rows)

    (
        alt.Chart(epochs_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("epochs:Q", scale=alt.Scale(type="log"), title="passes over the data"),
            y=alt.Y("inertia:Q", scale=alt.Scale(zero=False)),
            color="method:N",
        )
        .properties(width=600, height=300)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    sweep_epochs = mo.ui.slider(0.5, 5, step=0.5, value=2, label="n_epochs", show_value=True)
    sweep_seeds = mo.ui.slider(1, 5, value=1, label="seeds", show_value=True)
    mo.vstack(
        [
            mo.md(r"""
    ## Sweeping $\alpha$ and the batch size

    Every cell of the grid is a full `EMeans` fit with a fixed `alpha` and `batch_size`. The color is the inertia
    relative to the best cell in the grid (1 is the best). The black cross is what the scaling laws pick
    (`alpha=None, batch_size=None`).
    """),
            mo.hstack([sweep_epochs, sweep_seeds], justify="start"),
        ]
    )
    return sweep_epochs, sweep_seeds


@app.cell
def _(EMeans, X, emeans_alpha, emeans_batch_size, k, np, pl, sweep_epochs, sweep_seeds, time):
    n_epochs = sweep_epochs.value
    law_batch = emeans_batch_size(len(X), k, n_epochs, X.shape[1])
    law_alpha = emeans_alpha(len(X), k, n_epochs, law_batch)
    batch_sizes = [2**p for p in range(5, 16) if 2**p <= len(X) // 4]
    alphas = np.geomspace(0.01, 1.0, 9).round(4).tolist()

    _rows = []
    for _seed in range(sweep_seeds.value):
        for _batch in batch_sizes:
            for _alpha in alphas:
                _tic = time.perf_counter()
                _model = EMeans(k, n_epochs=n_epochs, batch_size=_batch, alpha=_alpha, random_state=_seed).fit(X)
                _rows.append(
                    {
                        "seed": _seed,
                        "batch_size": _batch,
                        "alpha": _alpha,
                        "inertia": _model.inertia_,
                        "seconds": time.perf_counter() - _tic,
                    }
                )
        _model = EMeans(k, n_epochs=n_epochs, random_state=_seed).fit(X)
        _rows.append(
            {"seed": _seed, "batch_size": law_batch, "alpha": law_alpha, "inertia": _model.inertia_, "seconds": None}
        )
    _raw = pl.DataFrame(_rows)
    law_inertia = _raw.filter(pl.col("seconds").is_null())["inertia"].mean()
    sweep = (
        _raw.filter(pl.col("seconds").is_not_null())
        .group_by("batch_size", "alpha")
        .agg(pl.col("inertia").mean(), pl.col("seconds").mean())
        .with_columns(relative_inertia=pl.col("inertia") / pl.col("inertia").min())
        .sort("batch_size", "alpha")
    )
    return law_alpha, law_batch, law_inertia, n_epochs, sweep


@app.cell
def _(alt, law_alpha, law_batch, law_inertia, mo, pl, sweep):
    _best = sweep.sort("inertia").row(0, named=True)
    _heat = (
        alt.Chart(sweep)
        .mark_rect()
        .encode(
            x=alt.X("alpha:O", title="alpha"),
            y=alt.Y("batch_size:O", sort="descending", title="batch size"),
            color=alt.Color(
                "relative_inertia:Q", scale=alt.Scale(scheme="viridis", reverse=True, domain=[1, 1.1], clamp=True)
            ),
            tooltip=["batch_size", "alpha", "inertia", "relative_inertia", "seconds"],
        )
        .properties(width=450, height=320, title="inertia / best inertia in the grid")
    )
    _lines = (
        alt.Chart(sweep)
        .mark_line(point=True)
        .encode(
            x=alt.X("alpha:Q", scale=alt.Scale(type="log")),
            y=alt.Y("relative_inertia:Q", scale=alt.Scale(domain=[1, 1.15], clamp=True)),
            color=alt.Color("batch_size:O", scale=alt.Scale(scheme="plasma")),
        )
        .properties(width=450, height=320, title="inertia / best, per batch size")
    )
    _law = (
        alt.Chart(pl.DataFrame({"alpha": [law_alpha], "relative_inertia": [law_inertia / sweep["inertia"].min()]}))
        .mark_point(shape="cross", size=200, color="black", filled=True)
        .encode(x="alpha:Q", y="relative_inertia:Q")
    )
    mo.vstack(
        [
            mo.hstack([_heat, _lines + _law]),
            mo.md(
                f"Best cell: `batch_size={_best['batch_size']}`, `alpha={_best['alpha']}`. "
                f"The scaling laws pick `batch_size={law_batch}`, `alpha={law_alpha:.3f}`, landing at "
                f"**{law_inertia / sweep['inertia'].min():.4f}x** the best inertia in the grid."
            ),
        ]
    )
    return


@app.cell
def _(alt, sweep):
    (
        alt.Chart(sweep.group_by("batch_size").agg(seconds=sweep["seconds"].mean()).sort("batch_size"))
        .mark_line(point=True)
        .encode(
            x=alt.X("batch_size:Q", scale=alt.Scale(type="log", base=2)),
            y=alt.Y("seconds:Q", scale=alt.Scale(type="log"), title="seconds per fit"),
        )
        .properties(width=450, height=220, title="fit time per batch size (pure NumPy, Python loop per batch)")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Can we do the same for Gaussian mixtures?

    e-means is an exponential moving average over the *sufficient statistics* of k-means: the mass $N_j$ and the sum
    $S_j$ of the rows assigned to cluster $j$. A Gaussian mixture has the same structure, with soft responsibilities
    $r_{ij}$ instead of hard assignments, plus second moments for the covariances. This makes the GMM version a
    stepwise / online EM (Cappé & Moulines, 2009) with e-means' choices bolted on: mass weighted updates, a cosine step
    schedule, the batch size and $\alpha$ scaling laws, and a final exact M-step on the statistics of the last pass.
    For every batch:

    $$
    n_j = \sum_i r_{ij}, \quad
    N_j \leftarrow (1-\alpha_t) N_j + \alpha_t n_j, \quad
    w = \frac{\alpha_t n_j}{N_j}, \quad
    \mu_j \leftarrow \mu_j + w\left(\tfrac{\sum_i r_{ij} x_i}{n_j} - \mu_j\right), \quad
    m_j \leftarrow m_j + w\left(\tfrac{\sum_i r_{ij} x_i^2}{n_j} - m_j\right)
    $$

    with diagonal variances $\sigma_j^2 = m_j - \mu_j^2$ and mixture weights $\pi_j \propto N_j$. Below is a prototype
    (diagonal covariances only) compared to scikit-learn's `GaussianMixture(covariance_type="diag")` on the held-out
    average log-likelihood. Both start from the same means.
    """)
    return


@app.cell
def _(emeans_alpha, emeans_batch_size, epoch_rows, logsumexp, math, np, scheduled_alpha):
    def diag_log_resp(X, weights, means, variances):
        precisions = 1.0 / variances
        quad = (X**2) @ precisions.T - 2 * X @ (means * precisions).T + (means**2 * precisions).sum(axis=1)
        log_prob = -0.5 * (X.shape[1] * np.log(2 * np.pi) + np.log(variances).sum(axis=1) + quad) + np.log(weights)
        log_norm = logsumexp(log_prob, axis=1)
        return log_prob - log_norm[:, None], log_norm

    def egmm(X, init_means, n_epochs=3.0, batch_size=None, alpha=None, reg_covar=1e-6, seed=0):
        """e-means style moving average EM for a diagonal Gaussian mixture. Returns (weights, means, variances)."""
        rng = np.random.default_rng(seed)
        X = X.astype(np.float64)
        n, d = X.shape
        k = len(init_means)
        batch_size = batch_size or emeans_batch_size(n, k, n_epochs, d)
        alpha = alpha or emeans_alpha(n, k, n_epochs, batch_size)
        means = np.array(init_means, dtype=np.float64)
        variances = np.tile(X.var(axis=0) + reg_covar, (k, 1))
        second = variances + means**2
        weights = np.full(k, 1.0 / k)
        mass = np.full(k, 1.0 / 65536)
        epochs = epoch_rows(n, n_epochs)
        total_steps = sum(math.ceil(rows / batch_size) for rows in epochs)
        step = 0
        for epoch, rows in enumerate(epochs):
            last = epoch == len(epochs) - 1
            pass0, pass1, pass2 = np.zeros(k), np.zeros((k, d)), np.zeros((k, d))
            order = rng.permutation(n)[:rows]
            for start in range(0, rows, batch_size):
                batch = X[order[start : start + batch_size]]
                resp = np.exp(diag_log_resp(batch, weights, means, variances)[0])
                n0, s1, s2 = resp.sum(axis=0), resp.T @ batch, resp.T @ batch**2
                if last:
                    pass0, pass1, pass2 = pass0 + n0, pass1 + s1, pass2 + s2
                a = scheduled_alpha(alpha, step, total_steps)
                mass = (1 - a) * mass + a * n0
                hit = n0 > 1e-10
                w = (a * n0[hit] / mass[hit])[:, None]
                means[hit] += w * (s1[hit] / n0[hit, None] - means[hit])
                second[hit] += w * (s2[hit] / n0[hit, None] - second[hit])
                variances = np.maximum(second - means**2, 0) + reg_covar
                weights = np.maximum(mass / mass.sum(), 1e-12)
                step += 1
        hit = pass0 > 1e-10
        means[hit] = pass1[hit] / pass0[hit, None]
        variances[hit] = np.maximum(pass2[hit] / pass0[hit, None] - means[hit] ** 2, 0) + reg_covar
        weights = (pass0 + 1e-10) / (pass0 + 1e-10).sum()
        return weights, means, variances

    def egmm_score(X, weights, means, variances):
        return float(diag_log_resp(X.astype(np.float64), weights, means, variances)[1].mean())

    return egmm, egmm_score


@app.cell(hide_code=True)
def _(mo):
    gmm_init = mo.ui.dropdown(["random rows", "k-means++"], value="random rows", label="starting means")
    gmm_k = mo.ui.slider(4, 128, step=4, value=32, label="n_components", show_value=True)
    mo.hstack([gmm_init, gmm_k], justify="start")
    return gmm_init, gmm_k


@app.cell
def _(gmm_init, gmm_k, kmeans_plusplus, make_blobs, np):
    # Anisotropic blobs of different widths: a setting where a diagonal GMM is the right model.
    _rng = np.random.default_rng(1)
    _X, _ = make_blobs(
        n_samples=60_000,
        centers=gmm_k.value,
        n_features=16,
        cluster_std=_rng.uniform(0.5, 2.5, gmm_k.value),
        random_state=0,
    )
    _X = _X * _rng.uniform(0.5, 2.0, 16)
    X_gmm_train, X_gmm_test = _X[:50_000], _X[50_000:]
    if gmm_init.value == "k-means++":
        gmm_means_init, _ = kmeans_plusplus(X_gmm_train, gmm_k.value, random_state=0)
    else:
        gmm_means_init = X_gmm_train[np.random.default_rng(0).choice(len(X_gmm_train), gmm_k.value, replace=False)]
    return X_gmm_test, X_gmm_train, gmm_means_init


@app.cell
def _(GaussianMixture, X_gmm_test, X_gmm_train, alt, egmm, egmm_score, gmm_means_init, pl, time):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    _rows = []
    for _iters in [1, 2, 3, 5, 10, 25, 50]:
        _tic = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _gmm = GaussianMixture(
                len(gmm_means_init),
                covariance_type="diag",
                means_init=gmm_means_init,
                max_iter=_iters,
                tol=0,
                random_state=0,
            ).fit(X_gmm_train)
        _rows.append(
            {
                "method": "EM (GaussianMixture)",
                "epochs": _iters,
                "test log-likelihood": _gmm.score(X_gmm_test),
                "seconds": time.perf_counter() - _tic,
            }
        )
    for _epochs in [0.5, 1, 1.5, 2, 3, 5]:
        _tic = time.perf_counter()
        _params = egmm(X_gmm_train, gmm_means_init, n_epochs=_epochs)
        _rows.append(
            {
                "method": "e-GMM (prototype)",
                "epochs": _epochs,
                "test log-likelihood": egmm_score(X_gmm_test, *_params),
                "seconds": time.perf_counter() - _tic,
            }
        )
    gmm_curve = pl.DataFrame(_rows)
    (
        alt.Chart(gmm_curve)
        .mark_line(point=True)
        .encode(
            x=alt.X("epochs:Q", scale=alt.Scale(type="log"), title="passes over the data"),
            y=alt.Y("test log-likelihood:Q", scale=alt.Scale(zero=False)),
            color="method:N",
            tooltip=["method", "epochs", "test log-likelihood", "seconds"],
        )
        .properties(width=600, height=300)
    )
    return


@app.cell
def _(X_gmm_test, X_gmm_train, alt, egmm, egmm_score, gmm_means_init, pl):
    _rows = []
    for _batch in [32, 128, 512, 2048, 8192]:
        for _alpha in [0.01, 0.03, 0.1, 0.3, 1.0]:
            _params = egmm(X_gmm_train, gmm_means_init, n_epochs=2, batch_size=_batch, alpha=_alpha)
            _rows.append(
                {"batch_size": _batch, "alpha": _alpha, "test log-likelihood": egmm_score(X_gmm_test, *_params)}
            )
    gmm_sweep = pl.DataFrame(_rows)
    _best = gmm_sweep["test log-likelihood"].max()
    (
        alt.Chart(gmm_sweep.with_columns(gap=_best - pl.col("test log-likelihood")))
        .mark_rect()
        .encode(
            x="alpha:O",
            y=alt.Y("batch_size:O", sort="descending"),
            color=alt.Color("gap:Q", scale=alt.Scale(scheme="viridis", reverse=False, domain=[0, 2], clamp=True)),
            tooltip=["batch_size", "alpha", "test log-likelihood"],
        )
        .properties(
            width=400, height=260, title="e-GMM, 2 epochs: gap to the best test log-likelihood (lower is better)"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the GMM prototype shows so far

    - Starting from random rows, the moving average version reaches a better held-out log-likelihood within one or
      two passes than full-batch EM reaches after 50 iterations. The noisy updates seem to help it escape the poor
      local optimum that EM gets stuck in.
    - Starting from k-means++ means the two end up close: EM converges fast from a good start, and e-GMM needs a few
      passes to catch up.
    - The k-means scaling laws do not transfer one to one: small batches combined with a large `alpha` make components
      collapse, and the best cell of the sweep sits at a larger batch than the law picks. A GMM version in the library
      would need its own (or a more conservative) default for the batch size and `alpha`.
    """)
    return


if __name__ == "__main__":
    app.run()
