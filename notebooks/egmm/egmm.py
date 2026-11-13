import math

import numpy as np
from scipy.special import logsumexp

from sklego.cluster.emeans import (
    _epoch_rows as epoch_rows,
)
from sklego.cluster.emeans import (
    _scheduled_alpha as scheduled_alpha,
)
from sklego.cluster.emeans import (
    emeans_alpha,
    emeans_batch_size,
)


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
