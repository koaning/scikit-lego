import math
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data

# Constants and scaling laws from `faiss/contrib/e_means/base.py`.
_ALPHA_MIN_FACTOR = 0.1
_CHECK_UNUSED_EVERY = 5
_INIT_USAGE = 1.0 / 65536.0
_SPLIT_EPSILON = 1e-4
_UNUSED_THRESHOLD = 0.1
_NO_REVIVAL_MAX_D = 2
_AUTO_QUALITY_MAX_D = 16

# batch_size = C * k^a * n^b * T^c
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}
# alpha0 = C * k^a * B^b * n^c * T^e
_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _round_power_of_two(value):
    if value <= 1.0:
        return 1
    low = 2.0 ** math.floor(math.log2(value))
    high = 2.0 * low
    return int(low) if value / low < high / value else int(high)


def emeans_batch_size(n_samples, n_clusters, n_epochs, n_features, batch_law="auto"):
    """Batch size picked by the e-means scaling law.

    Parameters
    ----------
    n_samples : int
        Number of rows in the training data.
    n_clusters : int
        Number of clusters.
    n_epochs : float
        Number of passes over the data.
    n_features : int
        Number of features, used to resolve `batch_law="auto"`.
    batch_law : Literal["auto", "throughput", "quality"], default="auto"
        `"quality"` gives batches about 8 times smaller than `"throughput"`, trading speed for a lower error.
        `"auto"` picks `"quality"` when `n_features <= 16` and `"throughput"` otherwise.

    Returns
    -------
    int
        The batch size.
    """
    if batch_law == "auto":
        batch_law = "quality" if n_features <= _AUTO_QUALITY_MAX_D else "throughput"
    c, b_k, b_n, b_t = _BATCH_LAWS[batch_law]
    batch = _round_power_of_two(c * n_clusters**b_k * n_samples**b_n * n_epochs**b_t)
    if batch_law == "quality":
        batch = min(batch, n_samples // 2)
    else:
        batch = min(batch, _round_power_of_two(n_samples / 2))
    return max(min(batch, n_samples), 1)


def emeans_alpha(n_samples, n_clusters, n_epochs, batch_size):
    """Starting step size picked by the e-means scaling law.

    Parameters
    ----------
    n_samples : int
        Number of rows in the training data.
    n_clusters : int
        Number of clusters.
    n_epochs : float
        Number of passes over the data.
    batch_size : int
        Rows per update.

    Returns
    -------
    float
        The starting step size, clipped to `[1e-4, 1]`.
    """
    c, a_k, a_b, a_n, a_t = _ALPHA_LAW
    alpha = c * n_clusters**a_k * batch_size**a_b * n_samples**a_n * n_epochs**a_t
    return float(np.clip(alpha, 1e-4, 1.0))


def _epoch_rows(n_samples, n_epochs):
    """Whole passes, then the fractional part, then a last full pass."""
    whole = math.floor(n_epochs)
    remainder = round((n_epochs - whole) * n_samples)
    if whole == 0:
        return [remainder]
    rows = [n_samples] * (whole - 1)
    if remainder > 0:
        rows.append(remainder)
    rows.append(n_samples)
    return rows


def _scheduled_alpha(alpha0, step, total_steps):
    """Cosine decay from `alpha0` down to a tenth of it."""
    fraction = min(step, total_steps) / max(total_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return alpha0 * (_ALPHA_MIN_FACTOR + (1.0 - _ALPHA_MIN_FACTOR) * cosine)


def _squared_distances(X, centers, center_norms):
    dist = np.einsum("ij,ij->i", X, X)[:, None] - 2.0 * (X @ centers.T) + center_norms[None, :]
    return np.maximum(dist, 0.0, out=dist)


def _nearest(X, centers):
    dist = _squared_distances(X, centers, np.einsum("ij,ij->i", centers, centers))
    labels = dist.argmin(axis=1)
    return labels, dist[np.arange(len(labels)), labels]


class EMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """e-means clustering: k-means trained with mini-batches and an exponential moving average of the centroids.

    Every mini-batch is assigned to its nearest centroids, after which each centroid moves towards the mean of its
    rows with a step that depends on the decaying step size `alpha` and on the mass of the cluster. The step size
    follows a cosine schedule from `alpha` down to `alpha / 10`. Clusters that receive (almost) no rows are revived
    by splitting a busy cluster. After the last pass every centroid is replaced by the exact mean of the rows it
    received during that pass.

    When `batch_size` and `alpha` are not given they are picked by the scaling laws from the paper, so that no
    hyperparameter sweep is needed. This is a NumPy port of `NumpyEmeans` from
    [faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means), see
    [e-means: outperforming k-means (in one-tenth the epochs)](https://arxiv.org/abs/2611.09501).

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    n_epochs : float, default=5.0
        Number of passes over the data, can be fractional.
    batch_size : int | None, default=None
        Rows per update. When `None` it is picked by the scaling law selected with `batch_law`.
    alpha : float | None, default=None
        Starting step size of the moving average, in `(0, 1]`. When `None` it is picked by the scaling law, which
        depends on the number of samples, `n_clusters`, `n_epochs` and the batch size.
    batch_law : Literal["auto", "throughput", "quality"], default="auto"
        Scaling law used to pick the batch size when `batch_size=None`. `"quality"` gives batches about 8 times
        smaller than `"throughput"`, trading speed for a lower error. `"auto"` picks `"quality"` when there are at
        most 16 features and `"throughput"` otherwise.
    init : Literal["random", "k-means++"] | array-like of shape (n_clusters, n_features), default="random"
        How to pick the starting centroids. `"random"` takes `n_clusters` random rows.
    unused_threshold : float | None, default=None
        Clusters whose mass falls below this fraction of the mean mass are revived by splitting a busy cluster. `0`
        turns revival off. When `None` it is `0.1`, or `0` when there are at most 2 features.
    random_state : int | np.random.RandomState | None, default=None
        Controls the starting centroids, the order of the rows and the revivals.

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        The centroids.
    labels_ : np.ndarray of shape (n_samples,)
        Index of the nearest centroid of each training row.
    inertia_ : float
        Sum of squared distances of the training rows to their nearest centroid.
    batch_size_ : int
        The batch size used during training.
    alpha_ : float
        The starting step size used during training.
    n_iter_ : int
        Number of passes over the data, a fractional pass counts as one.
    n_steps_ : int
        Number of mini-batch updates.
    n_splits_ : int
        Number of clusters revived during training.
    objective_history_ : list[float]
        Sum of squared distances measured during each pass, while the centroids move.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    from sklearn.datasets import make_blobs
    from sklego.cluster import EMeans

    X, _ = make_blobs(n_samples=10_000, centers=20, n_features=8, random_state=42)

    emeans = EMeans(n_clusters=20, n_epochs=2, random_state=42).fit(X)
    emeans.cluster_centers_.shape
    # (20, 8)
    emeans.batch_size_, round(emeans.alpha_, 3)
    # (32, 0.069)
    ```
    """

    def __init__(
        self,
        n_clusters=8,
        n_epochs=5.0,
        batch_size=None,
        alpha=None,
        batch_law="auto",
        init="random",
        unused_threshold=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.batch_law = batch_law
        self.init = init
        self.unused_threshold = unused_threshold
        self.random_state = random_state

    def _check_params(self, n_samples):
        if not isinstance(self.n_clusters, Integral) or self.n_clusters < 1:
            raise ValueError(f"n_clusters must be a positive integer, got {self.n_clusters}")
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}.")
        if not isinstance(self.n_epochs, Real) or not (0 < self.n_epochs < math.inf):
            raise ValueError(f"n_epochs must be a positive number, got {self.n_epochs}")
        if self.batch_size is not None and (not isinstance(self.batch_size, Integral) or self.batch_size < 1):
            raise ValueError(f"batch_size must be a positive integer or None, got {self.batch_size}")
        if self.alpha is not None and (not isinstance(self.alpha, Real) or not (0 < self.alpha <= 1)):
            raise ValueError(f"alpha must be in (0, 1] or None, got {self.alpha}")
        if self.batch_law not in ("auto", "throughput", "quality"):
            raise ValueError(f"batch_law must be one of 'auto', 'throughput' or 'quality', got {self.batch_law}")
        if self.unused_threshold is not None and (
            not isinstance(self.unused_threshold, Real) or not (0 <= self.unused_threshold < math.inf)
        ):
            raise ValueError(f"unused_threshold must be a non-negative number or None, got {self.unused_threshold}")

    def _init_centers(self, X, rng):
        if isinstance(self.init, str):
            if self.init == "random":
                return X[rng.choice(X.shape[0], size=self.n_clusters, replace=False)].copy()
            if self.init == "k-means++":
                centers, _ = kmeans_plusplus(X, self.n_clusters, random_state=rng)
                return centers.astype(X.dtype)
            raise ValueError(f"init must be 'random', 'k-means++' or an array, got {self.init}")
        centers = np.array(self.init, dtype=X.dtype)
        if centers.shape != (self.n_clusters, X.shape[1]):
            raise ValueError(
                f"init has shape {centers.shape}, expected (n_clusters, n_features) = {(self.n_clusters, X.shape[1])}"
            )
        return centers

    def fit(self, X, y=None):
        """Fit the e-means model on `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EMeans
            The fitted estimator.
        """
        X = validate_data(self, X=X, dtype=[np.float64, np.float32], reset=True)
        n, d = X.shape
        k = self.n_clusters
        self._check_params(n)
        rng = check_random_state(self.random_state)

        self.batch_size_ = (
            min(self.batch_size, n)
            if self.batch_size is not None
            else emeans_batch_size(n, k, self.n_epochs, d, self.batch_law)
        )
        self.alpha_ = (
            float(self.alpha) if self.alpha is not None else emeans_alpha(n, k, self.n_epochs, self.batch_size_)
        )
        threshold = self.unused_threshold
        if threshold is None:
            threshold = 0.0 if d <= _NO_REVIVAL_MAX_D else _UNUSED_THRESHOLD

        epoch_rows = _epoch_rows(n, self.n_epochs)
        if 0 in epoch_rows:
            raise ValueError("n_epochs is too small: the run would see no rows")
        total_steps = sum(math.ceil(rows / self.batch_size_) for rows in epoch_rows)

        centers = self._init_centers(X, rng)
        usage = np.full(k, _INIT_USAGE)
        step, n_splits = 0, 0
        self.objective_history_ = []

        for epoch, rows in enumerate(epoch_rows):
            last_pass = epoch == len(epoch_rows) - 1
            pass_counts = np.zeros(k)
            pass_sums = np.zeros((k, d)) if last_pass else None
            objective = 0.0
            order = rng.permutation(n)[:rows]
            for start in range(0, rows, self.batch_size_):
                batch = X[order[start : start + self.batch_size_]]
                labels, dist = _nearest(batch, centers)
                objective += float(dist.sum())
                one_hot = sparse.csr_matrix(
                    (np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(k, len(labels))
                )
                counts = np.bincount(labels, minlength=k).astype(np.float64)
                sums = np.asarray(one_hot @ batch.astype(np.float64))
                pass_counts += counts
                if last_pass:
                    pass_sums += sums

                alpha = _scheduled_alpha(self.alpha_, step, total_steps)
                centers, usage = self._apply_batch(centers, usage, counts, sums, alpha)
                if (step + 1) % _CHECK_UNUSED_EVERY == 0:
                    n_splits += self._revive_unused(centers, usage, threshold, rng)
                step += 1
            self.objective_history_.append(objective)

        occupied = pass_counts > 0
        centers[occupied] = (pass_sums[occupied] / pass_counts[occupied, None]).astype(X.dtype)

        self.cluster_centers_ = centers
        self.n_steps_ = step
        self.n_iter_ = len(epoch_rows)
        self.n_splits_ = n_splits
        self.labels_, dist = _nearest(X, centers)
        self.inertia_ = float(dist.sum())
        return self

    @staticmethod
    def _apply_batch(centers, usage, counts, sums, alpha):
        """Moving average update on centroids `c` and masses `N`, with `n_j` rows summing to `s_j` in the batch.

        ```text
        N_new = (1 - alpha) * N + alpha * n_j
        c_new = c + (alpha * n_j / N_new) * (s_j / n_j - c)
        ```

        A step of exactly 1 replaces the mass of occupied clusters and leaves the others alone.
        """
        if alpha < 1.0:
            usage = usage * (1.0 - alpha)
            retained = usage
        else:
            retained = np.zeros_like(usage)
        hit = counts > 0
        mass = retained[hit] + alpha * counts[hit]
        weight = (alpha * counts[hit] / mass)[:, None]
        means = sums[hit] / counts[hit, None]
        centers[hit] = (centers[hit] + weight * (means - centers[hit])).astype(centers.dtype)
        usage = usage.copy()
        usage[hit] = mass
        return centers, usage

    @staticmethod
    def _revive_unused(centers, usage, threshold, rng):
        """Split busy clusters, drawn in proportion to their mass, to replace the ones with almost no mass."""
        if threshold == 0.0:
            return 0
        limit = usage.sum() * threshold / len(usage)
        unused = np.flatnonzero(usage < limit)
        donors = np.flatnonzero((usage >= limit) & (usage > 0))
        n_split = min(len(unused), len(donors))
        if n_split == 0:
            return 0
        # Gumbel top-k: donors drawn without replacement in proportion to their mass.
        keys = np.log(usage[donors]) - np.log(-np.log(np.maximum(rng.random_sample(len(donors)), 1e-300)))
        sources = donors[np.argsort(-keys)[:n_split]]
        targets = unused[:n_split]
        offset = np.where(np.arange(centers.shape[1]) % 2 == 0, _SPLIT_EPSILON, -_SPLIT_EPSILON)
        base = centers[sources].copy()
        centers[sources] = base * (1 - offset)
        centers[targets] = base * (1 + offset)
        collapsed = np.all(centers[sources] == centers[targets], axis=1)
        centers[targets[collapsed], 0] = np.nextafter(centers[targets[collapsed], 0], np.inf)
        usage[sources] /= 2.0
        usage[targets] = usage[sources]
        return n_split

    def predict(self, X):
        """Index of the nearest centroid of each row in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to assign.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Index of the nearest centroid.
        """
        check_is_fitted(self, ["cluster_centers_"])
        X = validate_data(self, X=X, dtype=[np.float64, np.float32], reset=False)
        return _nearest(X, self.cluster_centers_.astype(X.dtype))[0]

    def transform(self, X):
        """Euclidean distance of each row in `X` to every centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_clusters)
            Distances to the centroids.
        """
        check_is_fitted(self, ["cluster_centers_"])
        X = validate_data(self, X=X, dtype=[np.float64, np.float32], reset=False)
        centers = self.cluster_centers_.astype(X.dtype)
        return np.sqrt(_squared_distances(X, centers, np.einsum("ij,ij->i", centers, centers)))

    def score(self, X, y=None):
        """Opposite of the sum of squared distances of the rows in `X` to their nearest centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to score.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        float
            The negative inertia, higher is better.
        """
        check_is_fitted(self, ["cluster_centers_"])
        X = validate_data(self, X=X, dtype=[np.float64, np.float32], reset=False)
        return -float(_nearest(X, self.cluster_centers_.astype(X.dtype))[1].sum())
