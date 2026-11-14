import math
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data

from sklego.cluster.emeans import (
    _BACKENDS,
    _CHECK_UNUSED_EVERY,
    _INIT_USAGE,
    _NO_REVIVAL_MAX_D,
    _UNUSED_THRESHOLD,
    EMeans,
    _alpha_schedule,
    _chunks,
    _epoch_rows,
    _round_power_of_two,
    _rust_module,
    _rust_seed,
    _scheduled_alpha,
    emeans_alpha,
    emeans_batch_size,
)

# The k-means scaling laws push alpha to 1 for longer runs, which turns the moving average into noisy minibatch EM.
# For mixtures we cap the step size and use batches large enough to give every component a few dozen rows.
_MAX_AUTO_ALPHA = 0.3
_ROWS_PER_COMPONENT = 64
_INIT_EMEANS_EPOCHS = 2.0
_MIN_MASS = 1e-10
_LOG_2PI = math.log(2.0 * math.pi)
# A revived component and its donor start this many standard deviations (Mahalanobis distance) from the donor's mean.
# Much smaller splits do not separate: two near-identical components share every row under soft responsibilities.
_SPLIT_DISTANCE = 1.0


def egmm_batch_size(n_samples, n_components, n_epochs, n_features):
    """Default batch size of `EGaussianMixture`.

    This is the e-means batch size, raised to at least 64 rows per component and rounded to a power of two.

    Parameters
    ----------
    n_samples : int
        Number of rows in the training data.
    n_components : int
        Number of mixture components.
    n_epochs : float
        Number of passes over the data.
    n_features : int
        Number of features.

    Returns
    -------
    int
        The batch size.
    """
    law = emeans_batch_size(n_samples, n_components, n_epochs, n_features)
    return max(min(max(law, _round_power_of_two(_ROWS_PER_COMPONENT * n_components)), n_samples), 1)


def egmm_alpha(n_samples, n_components, n_epochs, batch_size):
    """Default starting step size of `EGaussianMixture`: the e-means step size capped at 0.3.

    Parameters
    ----------
    n_samples : int
        Number of rows in the training data.
    n_components : int
        Number of mixture components.
    n_epochs : float
        Number of passes over the data.
    batch_size : int
        Rows per update.

    Returns
    -------
    float
        The starting step size.
    """
    return min(emeans_alpha(n_samples, n_components, n_epochs, batch_size), _MAX_AUTO_ALPHA)


def _features(X, covariance_type):
    """Columns whose responsibility-weighted sums are the sufficient statistics: `[x, x**2]` or `[x, |x|**2]`."""
    if covariance_type == "diag":
        return np.hstack([X, X * X])
    return np.hstack([X, np.einsum("ij,ij->i", X, X)[:, None]])


def _variances(means, second, covariance_type, reg_covar):
    """Variances from the means and the second moments kept by the moving average."""
    if covariance_type == "diag":
        return np.maximum(second - means**2, 0.0) + reg_covar
    return np.maximum(second[:, 0] - np.einsum("ij,ij->i", means, means), 0.0) / means.shape[1] + reg_covar


def _second_moments(means, variances, covariance_type):
    if covariance_type == "diag":
        return variances + means**2
    return (variances * means.shape[1] + np.einsum("ij,ij->i", means, means))[:, None]


def _log_prob_params(weights, means, variances, covariance_type):
    """Matrix `W` and offset `b` so that `_features(X) @ W + b` is the weighted log density of every component."""
    d = means.shape[1]
    precisions = 1.0 / variances
    if covariance_type == "diag":
        W = np.vstack([(means * precisions).T, -0.5 * precisions.T])
        log_det = np.log(variances).sum(axis=1)
        mahalanobis = (means**2 * precisions).sum(axis=1)
    else:
        W = np.vstack([(means * precisions[:, None]).T, -0.5 * precisions[None, :]])
        log_det = d * np.log(variances)
        mahalanobis = np.einsum("ij,ij->i", means, means) * precisions
    bias = np.log(weights) - 0.5 * (d * _LOG_2PI + log_det + mahalanobis)
    return W, bias


def _normalize_in_place(log_prob):
    """Turn weighted log densities into responsibilities, in place, and return the log density of each row."""
    log_max = log_prob.max(axis=1, keepdims=True)
    log_prob -= log_max
    np.exp(log_prob, out=log_prob)
    total = log_prob.sum(axis=1, keepdims=True)
    log_prob /= total
    return (log_max + np.log(total))[:, 0]


class EGaussianMixture(DensityMixin, BaseEstimator):
    """Gaussian mixture trained with mini-batches and an exponential moving average, the e-means way.

    This is EM with the M-step swapped for the moving average update of [`EMeans`][sklego.cluster.emeans.EMeans].
    Every mini-batch gets an E-step under the current parameters, and the responsibility-weighted sums of `x` and
    `x**2` then move each component's weight, mean and variance with a step size that follows a cosine decay from
    `alpha` down to `alpha / 10`. Components that are (almost) never the most likely one for any row are revived by
    splitting a busy component, as in `EMeans`. After the last pass the parameters are replaced by a full M-step on
    the sums gathered during that pass.

    By default the means start from an `EMeans` fit, so a few passes over the data reach the held-out likelihood
    that `sklearn.mixture.GaussianMixture` needs many full EM iterations for. Only diagonal and spherical
    covariances are supported.

    Parameters
    ----------
    n_components : int, default=1
        Number of mixture components.
    covariance_type : Literal["diag", "spherical"], default="diag"
        `"diag"` gives every component its own variance per feature, `"spherical"` one variance per component.
    n_epochs : float, default=5.0
        Number of passes over the data, can be fractional.
    batch_size : int | None, default=None
        Rows per update. When `None` it is the e-means scaling law batch size, raised to at least 64 rows per
        component.
    alpha : float | None, default=None
        Starting step size of the moving average, in `(0, 1]`. When `None` it is the e-means scaling law step size,
        capped at 0.3.
    init : Literal["e-means", "k-means++", "random"] | array-like of shape (n_components, n_features), default="e-means"
        How to start. `"e-means"` seeds with k-means++ on a subsample of `64 * n_components` rows, fits `EMeans` for
        two epochs and takes the weights, means and variances of its clusters. `"k-means++"` (on all rows),
        `"random"` (random rows) or an array of means start with equal weights and the variance of the whole
        dataset.
    reg_covar : float, default=1e-6
        Non-negative value added to the variances to keep them positive.
    unused_threshold : float | None, default=None
        Revival of unused components, as in `EMeans`. Every component keeps a moving average of the number of rows
        for which it is the most likely component, and components whose average falls below this fraction of the
        mean are revived by splitting a busy one. The donor is drawn in proportion to that average, and the two
        halves start on either side of its mean, one standard deviation away, with the donor's variances. `0` turns
        revival off. When `None` it is `0.1`, or `0` when there are at most 2 features.
    random_state : int | np.random.RandomState | None, default=None
        Controls the initialisation, the order of the rows and `sample`.
    backend : Literal["numpy", "rust"], default="numpy"
        `"rust"` runs the training loop (and the `EMeans` initialisation) and the density evaluations
        (`score_samples`, `score`, `predict`, `predict_proba`) in compiled Rust, which needs the `sklego-rust` package
        (`pip install ./rust` from the scikit-lego repository). Both backends run the same algorithm with the same
        defaults, but they draw their random numbers differently, so a given `random_state` gives a different
        (equally good) model in each.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_components,)
        The weight of each component.
    means_ : np.ndarray of shape (n_components, n_features)
        The mean of each component.
    covariances_ : np.ndarray
        The variances, of shape `(n_components, n_features)` for `"diag"` and `(n_components,)` for `"spherical"`,
        the same layout as `sklearn.mixture.GaussianMixture`.
    precisions_ : np.ndarray
        The inverse of `covariances_`.
    precisions_cholesky_ : np.ndarray
        The square root of `precisions_`.
    lower_bound_ : float
        Average log-likelihood per row measured during the last pass, while the parameters move.
    batch_size_ : int
        The batch size used during training.
    alpha_ : float
        The starting step size used during training.
    n_iter_ : int
        Number of passes over the data, a fractional pass counts as one.
    n_steps_ : int
        Number of mini-batch updates.
    n_splits_ : int
        Number of components revived during training.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    from sklearn.datasets import make_blobs
    from sklego.mixture import EGaussianMixture

    X, _ = make_blobs(n_samples=10_000, centers=20, n_features=8, random_state=42)

    gmm = EGaussianMixture(n_components=20, random_state=42).fit(X)
    gmm.means_.shape
    # (20, 8)
    gmm.batch_size_, gmm.alpha_
    # (1024, 0.3)
    samples, labels = gmm.sample(5)
    ```
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="diag",
        n_epochs=5.0,
        batch_size=None,
        alpha=None,
        init="e-means",
        reg_covar=1e-6,
        unused_threshold=None,
        random_state=None,
        backend="numpy",
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.init = init
        self.reg_covar = reg_covar
        self.unused_threshold = unused_threshold
        self.random_state = random_state
        self.backend = backend

    def _check_params(self, n_samples):
        if not isinstance(self.n_components, Integral) or self.n_components < 1:
            raise ValueError(f"n_components must be a positive integer, got {self.n_components}")
        if n_samples < self.n_components:
            raise ValueError(f"n_samples={n_samples} should be >= n_components={self.n_components}.")
        if self.covariance_type not in ("diag", "spherical"):
            raise ValueError(f"covariance_type must be 'diag' or 'spherical', got {self.covariance_type}")
        if not isinstance(self.n_epochs, Real) or not (0 < self.n_epochs < math.inf):
            raise ValueError(f"n_epochs must be a positive number, got {self.n_epochs}")
        if self.batch_size is not None and (not isinstance(self.batch_size, Integral) or self.batch_size < 1):
            raise ValueError(f"batch_size must be a positive integer or None, got {self.batch_size}")
        if self.alpha is not None and (not isinstance(self.alpha, Real) or not (0 < self.alpha <= 1)):
            raise ValueError(f"alpha must be in (0, 1] or None, got {self.alpha}")
        if not isinstance(self.reg_covar, Real) or not (0 <= self.reg_covar < math.inf):
            raise ValueError(f"reg_covar must be a non-negative number, got {self.reg_covar}")
        if self.unused_threshold is not None and (
            not isinstance(self.unused_threshold, Real) or not (0 <= self.unused_threshold < math.inf)
        ):
            raise ValueError(f"unused_threshold must be a non-negative number or None, got {self.unused_threshold}")
        if isinstance(self.init, str) and self.init not in ("e-means", "k-means++", "random"):
            raise ValueError(f"init must be 'e-means', 'k-means++', 'random' or an array, got {self.init}")
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend must be 'numpy' or 'rust', got {self.backend}")

    def _initialize(self, X, shift, rng):
        """Weights, (shifted) means and variances to start from."""
        n, d = X.shape
        k = self.n_components
        global_var = X.var(axis=0) + self.reg_covar
        if not isinstance(self.init, str):
            means = np.array(self.init, dtype=np.float64)
            if means.shape != (k, d):
                raise ValueError(f"init has shape {means.shape}, expected (n_components, n_features) = {(k, d)}")
            means = means - shift
        elif self.init == "random":
            means = X[rng.choice(n, size=k, replace=False)] - shift
        elif self.init == "k-means++":
            means = kmeans_plusplus(X, k, random_state=rng)[0] - shift
        else:
            return self._initialize_from_emeans(X, shift, rng, global_var)
        variances = np.tile(global_var, (k, 1))
        if self.covariance_type == "spherical":
            variances = variances.mean(axis=1)
        return np.full(k, 1.0 / k), means, variances

    def _initialize_from_emeans(self, X, shift, rng, global_var):
        n, d = X.shape
        k = self.n_components
        # k-means++ on a subsample: random seeds often put two centroids in one well separated cluster, and
        # k-means++ on all rows costs more than the rest of the fit.
        subsample = X[rng.choice(n, size=min(n, _ROWS_PER_COMPONENT * k), replace=False)]
        seeds = kmeans_plusplus(subsample, k, random_state=rng)[0]
        emeans = EMeans(
            n_clusters=k, n_epochs=_INIT_EMEANS_EPOCHS, init=seeds, random_state=rng, backend=self.backend
        ).fit(X)
        counts = np.bincount(emeans.labels_, minlength=k).astype(np.float64)
        if self.backend == "rust":
            sums = _rust_module().egmm_label_stats(
                X, shift, emeans.labels_.astype(np.int64, copy=False), k, self.covariance_type
            )
        else:
            sums = np.zeros((k, d + (d if self.covariance_type == "diag" else 1)))
            for rows in _chunks(X.shape[0], k):
                one_hot = np.zeros((rows.stop - rows.start, k))
                one_hot[np.arange(rows.stop - rows.start), emeans.labels_[rows]] = 1.0
                sums += one_hot.T @ _features(X[rows] - shift, self.covariance_type)
        means = emeans.cluster_centers_.astype(np.float64) - shift
        variances = np.tile(global_var, (k, 1))
        if self.covariance_type == "spherical":
            variances = variances.mean(axis=1)
        hit = counts > 0
        means[hit] = sums[hit, :d] / counts[hit, None]
        variances[hit] = _variances(means[hit], sums[hit, d:] / counts[hit, None], self.covariance_type, self.reg_covar)
        weights = (counts + _MIN_MASS) / (counts + _MIN_MASS).sum()
        return weights, means, variances

    def fit(self, X, y=None):
        """Fit the mixture on `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EGaussianMixture
            The fitted estimator.
        """
        X = validate_data(self, X=X, dtype=np.float64, reset=True)
        if self.backend == "rust":
            X = np.ascontiguousarray(X)
        n, d = X.shape
        k = self.n_components
        self._check_params(n)
        rng = check_random_state(self.random_state)

        self.batch_size_ = (
            min(self.batch_size, n) if self.batch_size is not None else egmm_batch_size(n, k, self.n_epochs, d)
        )
        self.alpha_ = float(self.alpha) if self.alpha is not None else egmm_alpha(n, k, self.n_epochs, self.batch_size_)
        epoch_rows = _epoch_rows(n, self.n_epochs)
        if 0 in epoch_rows:
            raise ValueError("n_epochs is too small: the run would see no rows")
        total_steps = sum(math.ceil(rows / self.batch_size_) for rows in epoch_rows)
        threshold = self.unused_threshold
        if threshold is None:
            threshold = 0.0 if d <= _NO_REVIVAL_MAX_D else _UNUSED_THRESHOLD

        # Work on data centred around its mean, the variances come from E[x**2] - E[x]**2.
        shift = X.mean(axis=0)
        weights, means, variances = self._initialize(X, shift, rng)
        if self.backend == "rust":
            return self._fit_rust(X, shift, weights, means, variances, epoch_rows, total_steps, threshold, rng)

        second = _second_moments(means, variances, self.covariance_type)
        mass = np.full(k, _INIT_USAGE)
        usage = np.full(k, _INIT_USAGE)
        step, n_splits = 0, 0

        for epoch, rows in enumerate(epoch_rows):
            last_pass = epoch == len(epoch_rows) - 1
            pass_mass, pass_sums, log_likelihood = np.zeros(k), np.zeros((k, second.shape[1] + d)), 0.0
            order = rng.permutation(n)[:rows]
            for start in range(0, rows, self.batch_size_):
                Z = _features(X[order[start : start + self.batch_size_]] - shift, self.covariance_type)
                W, bias = _log_prob_params(weights, means, variances, self.covariance_type)
                resp = Z @ W
                resp += bias
                log_norm = _normalize_in_place(resp)
                batch_mass, batch_sums = resp.sum(axis=0), resp.T @ Z
                if last_pass:
                    pass_mass += batch_mass
                    pass_sums += batch_sums
                    log_likelihood += float(log_norm.sum())

                alpha = _scheduled_alpha(self.alpha_, step, total_steps)
                mass = (1.0 - alpha) * mass + alpha * batch_mass
                usage = (1.0 - alpha) * usage + alpha * np.bincount(resp.argmax(axis=1), minlength=k)
                hit = batch_mass > _MIN_MASS
                step_size = (alpha * batch_mass[hit] / mass[hit])[:, None]
                batch_means = batch_sums[hit] / batch_mass[hit, None]
                means[hit] += step_size * (batch_means[:, :d] - means[hit])
                second[hit] += step_size * (batch_means[:, d:] - second[hit])
                variances = _variances(means, second, self.covariance_type, self.reg_covar)
                if (step + 1) % _CHECK_UNUSED_EVERY == 0:
                    n_splits += self._revive_unused(means, second, variances, mass, usage, threshold, rng)
                weights = np.maximum(mass / mass.sum(), _MIN_MASS)
                weights /= weights.sum()
                step += 1

        hit = pass_mass > _MIN_MASS
        means[hit] = pass_sums[hit, :d] / pass_mass[hit, None]
        second[hit] = pass_sums[hit, d:] / pass_mass[hit, None]

        self.weights_ = (pass_mass + _MIN_MASS) / (pass_mass + _MIN_MASS).sum()
        self.means_ = means + shift
        self.covariances_ = _variances(means, second, self.covariance_type, self.reg_covar)
        self.precisions_ = 1.0 / self.covariances_
        self.precisions_cholesky_ = np.sqrt(self.precisions_)
        self.lower_bound_ = log_likelihood / epoch_rows[-1]
        self.n_iter_ = len(epoch_rows)
        self.n_steps_ = step
        self.n_splits_ = n_splits
        return self

    def _revive_unused(self, means, second, variances, mass, usage, threshold, rng):
        """Split busy components to replace the ones that are (almost) never the most likely component of a row.

        `usage` is the moving average of the number of rows for which each component is the most likely one, as the
        cluster mass in `EMeans`. The soft responsibility mass is no use here: components that share their rows
        (say all of them start at the same point) keep an equal mass forever. Works in place on the (shifted)
        means, the second moments, the variances, the masses and `usage`.
        """
        if threshold == 0.0:
            return 0
        k, d = means.shape
        limit = usage.sum() * threshold / k
        unused = np.flatnonzero(usage < limit)
        donors = np.flatnonzero((usage >= limit) & (usage > 0))
        n_split = min(len(unused), len(donors))
        if n_split == 0:
            return 0
        # Gumbel top-k: donors drawn without replacement in proportion to their mass.
        keys = np.log(usage[donors]) - np.log(-np.log(np.maximum(rng.random_sample(len(donors)), 1e-300)))
        sources = donors[np.argsort(-keys)[:n_split]]
        targets = unused[:n_split]
        # Move both halves along a fixed sign pattern, `_SPLIT_DISTANCE` standard deviations from the donor's mean.
        signs = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
        scale = np.sqrt(variances[sources])
        if self.covariance_type == "spherical":
            scale = scale[:, None]
        offset = (_SPLIT_DISTANCE / math.sqrt(d)) * scale * signs
        base = means[sources].copy()
        means[sources] = base - offset
        means[targets] = base + offset
        variances[targets] = variances[sources]
        changed = np.concatenate([sources, targets])
        second[changed] = _second_moments(means[changed], variances[changed], self.covariance_type)
        mass[targets] = mass[sources] = (mass[sources] + mass[targets]) / 2.0
        usage[sources] /= 2.0
        usage[targets] = usage[sources]
        return n_split

    def _fit_rust(self, X, shift, weights, means, variances, epoch_rows, total_steps, threshold, rng):
        weights, means, covariances, lower_bound, n_steps, n_splits = _rust_module().egmm_fit(
            X,
            shift,
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(means, dtype=np.float64),
            np.ascontiguousarray(variances, dtype=np.float64),
            self.covariance_type,
            epoch_rows,
            self.batch_size_,
            _alpha_schedule(self.alpha_, total_steps),
            float(self.reg_covar),
            float(threshold),
            _rust_seed(rng),
        )
        self.weights_ = weights
        self.means_ = means + shift
        self.covariances_ = covariances
        self.precisions_ = 1.0 / self.covariances_
        self.precisions_cholesky_ = np.sqrt(self.precisions_)
        self.lower_bound_ = float(lower_bound)
        self.n_iter_ = len(epoch_rows)
        self.n_steps_ = n_steps
        self.n_splits_ = n_splits
        return self

    def _evaluate_rust(self, X, output):
        """`score_samples`, `predict_proba` or `predict` on the Rust backend."""
        check_is_fitted(self, ["weights_", "means_", "covariances_"])
        X = np.ascontiguousarray(validate_data(self, X=X, dtype=np.float64, reset=False))
        shift = self.weights_ @ self.means_
        return _rust_module().egmm_evaluate(
            X,
            shift,
            np.ascontiguousarray(self.weights_, dtype=np.float64),
            np.ascontiguousarray(self.means_ - shift, dtype=np.float64),
            np.ascontiguousarray(self.covariances_, dtype=np.float64),
            self.covariance_type,
            output,
        )

    def _weighted_log_prob(self, X):
        """`log(weight_j) + log N(x | component j)` for every row of `X`, computed by `_chunks` of rows.

        Returns the number of rows and a generator of `(rows, log_prob)` pairs.
        """
        check_is_fitted(self, ["weights_", "means_", "covariances_"])
        X = validate_data(self, X=X, dtype=np.float64, reset=False)
        # Shift both data and means by the same point so that the expanded square stays accurate.
        shift = self.weights_ @ self.means_
        W, bias = _log_prob_params(self.weights_, self.means_ - shift, self.covariances_, self.covariance_type)

        def blocks():
            for rows in _chunks(X.shape[0], self.n_components + 2 * X.shape[1]):
                log_prob = _features(X[rows] - shift, self.covariance_type) @ W
                log_prob += bias
                yield rows, log_prob

        return X.shape[0], blocks()

    def score_samples(self, X):
        """Log-likelihood of each row in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to score.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Log density of every row.
        """
        if self.backend == "rust":
            return self._evaluate_rust(X, "score_samples")
        n, blocks = self._weighted_log_prob(X)
        out = np.empty(n)
        for rows, log_prob in blocks:
            out[rows] = _normalize_in_place(log_prob)
        return out

    def score(self, X, y=None):
        """Average log-likelihood of the rows in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to score.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        float
            Mean log density, higher is better.
        """
        return float(self.score_samples(X).mean())

    def predict_proba(self, X):
        """Posterior probability of every component for each row in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to assign.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Responsibilities, every row sums to one.
        """
        if self.backend == "rust":
            return self._evaluate_rust(X, "predict_proba")
        n, blocks = self._weighted_log_prob(X)
        out = np.empty((n, self.n_components))
        for rows, log_prob in blocks:
            _normalize_in_place(log_prob)
            out[rows] = log_prob
        return out

    def predict(self, X):
        """Most likely component of each row in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to assign.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Component labels.
        """
        if self.backend == "rust":
            return self._evaluate_rust(X, "predict").astype(np.intp, copy=False)
        n, blocks = self._weighted_log_prob(X)
        out = np.empty(n, dtype=np.intp)
        for rows, log_prob in blocks:
            out[rows] = log_prob.argmax(axis=1)
        return out

    def fit_predict(self, X, y=None):
        """Fit the mixture on `X` and return the most likely component of each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Component labels.
        """
        return self.fit(X).predict(X)

    def sample(self, n_samples=1):
        """Draw rows from the fitted mixture.

        Parameters
        ----------
        n_samples : int, default=1
            Number of rows to draw.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            The drawn rows.
        y : np.ndarray of shape (n_samples,)
            The component each row was drawn from.
        """
        check_is_fitted(self, ["weights_", "means_", "covariances_"])
        if not isinstance(n_samples, Integral) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
        rng = check_random_state(self.random_state)
        labels = np.repeat(np.arange(self.n_components), rng.multinomial(n_samples, self.weights_))
        scale = np.sqrt(self.covariances_)
        scale = scale[labels] if self.covariance_type == "diag" else scale[labels, None]
        return self.means_[labels] + rng.standard_normal((n_samples, self.n_features_in_)) * scale, labels

    def _n_parameters(self):
        k, d = self.n_components, self.n_features_in_
        covariance_params = k * d if self.covariance_type == "diag" else k
        return covariance_params + k * d + k - 1

    def bic(self, X):
        """Bayesian information criterion on `X`, lower is better.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to evaluate.

        Returns
        -------
        float
            The BIC.
        """
        n = len(X)
        return -2.0 * self.score(X) * n + self._n_parameters() * np.log(n)

    def aic(self, X):
        """Akaike information criterion on `X`, lower is better.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to evaluate.

        Returns
        -------
        float
            The AIC.
        """
        return -2.0 * self.score(X) * len(X) + 2.0 * self._n_parameters()
