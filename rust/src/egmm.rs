//! e-GMM: a port of the training loop and the density evaluation of `sklego.mixture.EGaussianMixture`.
//!
//! The data is used shifted by `shift` (the training mean, or the mixture mean when scoring), and the shifted rows
//! are expanded on the fly into the features `z = [x, x**2]` (diagonal) or `z = [x, |x|**2]` (spherical). The
//! weighted log density of every component is then `z @ W + bias`, and the sufficient statistics are `resp.T @ z`.

use rayon::prelude::*;

use crate::emeans::{block_rows, MIN_ROWS_PER_TASK, PARALLEL_MIN};
use crate::linalg::{add_matmul_at, matmul, Rng};

const INIT_USAGE: f64 = 1.0 / 65536.0;
const MIN_MASS: f64 = 1e-10;
const LOG_2PI: f64 = 1.837_877_066_409_345_3;
const CHECK_UNUSED_EVERY: usize = 5;
/// See `_SPLIT_DISTANCE` in `sklego/mixture/egmm.py`.
const SPLIT_DISTANCE: f64 = 1.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Covariance {
    Diag,
    Spherical,
}

impl Covariance {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "diag" => Some(Covariance::Diag),
            "spherical" => Some(Covariance::Spherical),
            _ => None,
        }
    }

    /// Number of second-moment columns: one per feature, or one in total.
    pub fn width(self, d: usize) -> usize {
        match self {
            Covariance::Diag => d,
            Covariance::Spherical => 1,
        }
    }
}

/// Shapes shared by every function: `d` features, `k` components, `s` second-moment columns.
#[derive(Clone, Copy)]
pub struct Dims {
    pub d: usize,
    pub k: usize,
    pub s: usize,
    pub cov: Covariance,
}

impl Dims {
    pub fn new(d: usize, k: usize, cov: Covariance) -> Self {
        Dims {
            d,
            k,
            s: cov.width(d),
            cov,
        }
    }

    fn z(&self) -> usize {
        self.d + self.s
    }
}

/// The matrix `W` (z x k) and offset `bias` (k) of `_log_prob_params`.
pub fn log_prob_params(
    weights: &[f64],
    means: &[f64],
    variances: &[f64],
    dims: Dims,
) -> (Vec<f64>, Vec<f64>) {
    let Dims { d, k, s, cov } = dims;
    let mut w = vec![0.0; (d + s) * k];
    let mut bias = vec![0.0; k];
    for j in 0..k {
        let mean = &means[j * d..(j + 1) * d];
        let (log_det, mahalanobis) = match cov {
            Covariance::Diag => {
                let var = &variances[j * d..(j + 1) * d];
                let (mut log_det, mut mahalanobis) = (0.0, 0.0);
                for p in 0..d {
                    let precision = 1.0 / var[p];
                    w[p * k + j] = mean[p] * precision;
                    w[(d + p) * k + j] = -0.5 * precision;
                    log_det += var[p].ln();
                    mahalanobis += mean[p] * mean[p] * precision;
                }
                (log_det, mahalanobis)
            }
            Covariance::Spherical => {
                let precision = 1.0 / variances[j];
                for p in 0..d {
                    w[p * k + j] = mean[p] * precision;
                }
                w[d * k + j] = -0.5 * precision;
                (
                    d as f64 * variances[j].ln(),
                    mean.iter().map(|m| m * m).sum::<f64>() * precision,
                )
            }
        };
        bias[j] = weights[j].ln() - 0.5 * (d as f64 * LOG_2PI + log_det + mahalanobis);
    }
    (w, bias)
}

/// Variances from the means and the second moments, see `_variances`.
fn variances_from(means: &[f64], second: &[f64], reg_covar: f64, dims: Dims, out: &mut [f64]) {
    let Dims { d, k, cov, .. } = dims;
    for j in 0..k {
        let mean = &means[j * d..(j + 1) * d];
        match cov {
            Covariance::Diag => {
                for p in 0..d {
                    out[j * d + p] = (second[j * d + p] - mean[p] * mean[p]).max(0.0) + reg_covar;
                }
            }
            Covariance::Spherical => {
                let norm: f64 = mean.iter().map(|m| m * m).sum();
                out[j] = (second[j] - norm).max(0.0) / d as f64 + reg_covar;
            }
        }
    }
}

/// Second moments from means and variances, see `_second_moments`.
fn second_moments(means: &[f64], variances: &[f64], dims: Dims) -> Vec<f64> {
    let Dims { d, k, s, cov } = dims;
    let mut second = vec![0.0; k * s];
    for j in 0..k {
        let mean = &means[j * d..(j + 1) * d];
        match cov {
            Covariance::Diag => {
                for p in 0..d {
                    second[j * d + p] = variances[j * d + p] + mean[p] * mean[p];
                }
            }
            Covariance::Spherical => {
                second[j] = variances[j] * d as f64 + mean.iter().map(|m| m * m).sum::<f64>();
            }
        }
    }
    second
}

/// Write the features of the shifted row `x - shift` into `z`.
#[inline]
fn features(row: &[f64], shift: &[f64], cov: Covariance, z: &mut [f64]) {
    let d = row.len();
    let mut norm = 0.0;
    for p in 0..d {
        let v = row[p] - shift[p];
        z[p] = v;
        match cov {
            Covariance::Diag => z[d + p] = v * v,
            Covariance::Spherical => norm += v * v,
        }
    }
    if cov == Covariance::Spherical {
        z[d] = norm;
    }
}

/// Weighted log densities to responsibilities, in place. Returns the log density of the row.
#[inline]
fn normalize(row: &mut [f64]) -> f64 {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        total += *v;
    }
    for v in row.iter_mut() {
        *v /= total;
    }
    max + total.ln()
}

/// Responsibility mass, `resp.T @ z`, summed log density and the number of rows for which each component is the
/// most likely one, of a set of rows.
pub struct Accum {
    mass: Vec<f64>,
    sums: Vec<f64>,
    log_likelihood: f64,
    most_likely: Vec<f64>,
}

impl Accum {
    fn new(dims: Dims) -> Self {
        Accum {
            mass: vec![0.0; dims.k],
            sums: vec![0.0; dims.k * dims.z()],
            log_likelihood: 0.0,
            most_likely: vec![0.0; dims.k],
        }
    }

    fn reset(&mut self) {
        self.mass.fill(0.0);
        self.sums.fill(0.0);
        self.log_likelihood = 0.0;
        self.most_likely.fill(0.0);
    }

    fn merge(&mut self, other: &Accum) {
        for (a, b) in self.mass.iter_mut().zip(&other.mass) {
            *a += b;
        }
        for (a, b) in self.sums.iter_mut().zip(&other.sums) {
            *a += b;
        }
        self.log_likelihood += other.log_likelihood;
        for (a, b) in self.most_likely.iter_mut().zip(&other.most_likely) {
            *a += b;
        }
    }
}

/// Index of the largest value, the first one on ties (like `np.argmax`).
#[inline]
fn argmax(row: &[f64]) -> usize {
    let mut best = 0;
    for j in 1..row.len() {
        if row[j] > row[best] {
            best = j;
        }
    }
    best
}

/// Rows to evaluate: picked by index (a mini-batch) or a contiguous range (a full-data pass).
#[derive(Clone, Copy)]
enum Rows<'a> {
    Index(&'a [usize]),
    Range(usize, usize),
}

impl Rows<'_> {
    fn len(&self) -> usize {
        match self {
            Rows::Index(index) => index.len(),
            Rows::Range(start, stop) => stop - start,
        }
    }

    #[inline]
    fn get(&self, r: usize) -> usize {
        match self {
            Rows::Index(index) => index[r],
            Rows::Range(start, _) => start + r,
        }
    }
}

/// Buffers reused across blocks: the features `z`, the log densities or responsibilities, and per-row results.
#[derive(Default)]
struct Block {
    z: Vec<f64>,
    resp: Vec<f64>,
    log_norm: Vec<f64>,
    labels: Vec<usize>,
    ones: Vec<f64>,
}

impl Block {
    /// Fill `z` with the features of `rows` and `resp` with `z @ W + bias`. With `normalize`, `resp` becomes the
    /// responsibilities, `log_norm` the log density and `labels` the most likely component of every row. The work outside the GEMM is spread over
    /// threads when the block is large.
    #[allow(clippy::too_many_arguments)]
    fn log_prob(
        &mut self,
        x: &[f64],
        rows: Rows,
        shift: &[f64],
        w: &[f64],
        bias: &[f64],
        dims: Dims,
        normalize_rows: bool,
    ) {
        let (d, k, zd) = (dims.d, dims.k, dims.z());
        let b = rows.len();
        self.z.resize(b * zd, 0.0);
        let fill = |(r, z): (usize, &mut [f64])| {
            let i = rows.get(r);
            features(&x[i * d..(i + 1) * d], shift, dims.cov, z)
        };
        if b * zd < PARALLEL_MIN {
            self.z.chunks_exact_mut(zd).enumerate().for_each(fill);
        } else {
            self.z
                .par_chunks_exact_mut(zd)
                .enumerate()
                .with_min_len(MIN_ROWS_PER_TASK)
                .for_each(fill);
        }
        self.resp.resize(b * k, 0.0);
        matmul(&self.z, w, b, zd, k, &mut self.resp);
        self.log_norm.resize(b, 0.0);
        self.labels.resize(b, 0);
        let finish = |((row, log_norm), label): ((&mut [f64], &mut f64), &mut usize)| {
            for (v, &bj) in row.iter_mut().zip(bias) {
                *v += bj;
            }
            if normalize_rows {
                *log_norm = normalize(row);
                *label = argmax(row);
            }
        };
        if b * k < PARALLEL_MIN {
            self.resp
                .chunks_exact_mut(k)
                .zip(self.log_norm.iter_mut())
                .zip(self.labels.iter_mut())
                .for_each(finish);
        } else {
            self.resp
                .par_chunks_exact_mut(k)
                .zip(self.log_norm.par_iter_mut())
                .zip(self.labels.par_iter_mut())
                .with_min_len(MIN_ROWS_PER_TASK)
                .for_each(finish);
        }
    }

    /// E-step on the rows `index`: their responsibility mass, `resp.T @ z` and summed log density, in `acc`.
    #[allow(clippy::too_many_arguments)]
    fn estep(
        &mut self,
        x: &[f64],
        index: &[usize],
        shift: &[f64],
        w: &[f64],
        bias: &[f64],
        dims: Dims,
        acc: &mut Accum,
    ) {
        let b = index.len();
        self.log_prob(x, Rows::Index(index), shift, w, bias, dims, true);
        acc.reset();
        acc.log_likelihood = self.log_norm.iter().sum();
        for &label in &self.labels {
            acc.most_likely[label] += 1.0;
        }
        self.ones.resize(b, 1.0);
        add_matmul_at(&self.resp, &self.ones, b, dims.k, 1, &mut acc.mass);
        add_matmul_at(&self.resp, &self.z, b, dims.k, dims.z(), &mut acc.sums);
    }
}

/// Split busy components to replace the ones that are (almost) never the most likely component of a row, in place.
/// See `EGaussianMixture._revive_unused`.
#[allow(clippy::too_many_arguments)]
pub fn revive_unused(
    means: &mut [f64],
    second: &mut [f64],
    variances: &mut [f64],
    mass: &mut [f64],
    usage: &mut [f64],
    dims: Dims,
    threshold: f64,
    rng: &mut Rng,
) -> usize {
    if threshold == 0.0 {
        return 0;
    }
    let Dims { d, k, s, cov } = dims;
    let limit = usage.iter().sum::<f64>() * threshold / k as f64;
    let unused: Vec<usize> = (0..k).filter(|&j| usage[j] < limit).collect();
    let donors: Vec<usize> = (0..k)
        .filter(|&j| usage[j] >= limit && usage[j] > 0.0)
        .collect();
    let n_split = unused.len().min(donors.len());
    if n_split == 0 {
        return 0;
    }
    // Gumbel top-k: donors drawn without replacement in proportion to their usage.
    let mut keyed: Vec<(f64, usize)> = donors
        .iter()
        .map(|&j| (usage[j].ln() - (-rng.uniform().max(1e-300).ln()).ln(), j))
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
    let factor = SPLIT_DISTANCE / (d as f64).sqrt();
    for (t, &(_, source)) in keyed.iter().take(n_split).enumerate() {
        let target = unused[t];
        for p in 0..d {
            let std = match cov {
                Covariance::Diag => variances[source * d + p].sqrt(),
                Covariance::Spherical => variances[source].sqrt(),
            };
            let offset = factor * std * if p % 2 == 0 { 1.0 } else { -1.0 };
            let base = means[source * d + p];
            means[source * d + p] = base - offset;
            means[target * d + p] = base + offset;
        }
        for q in 0..s {
            variances[target * s + q] = variances[source * s + q];
        }
        for j in [source, target] {
            let mean = &means[j * d..(j + 1) * d];
            match cov {
                Covariance::Diag => {
                    for p in 0..d {
                        second[j * d + p] = variances[j * d + p] + mean[p] * mean[p];
                    }
                }
                Covariance::Spherical => {
                    second[j] = variances[j] * d as f64 + mean.iter().map(|m| m * m).sum::<f64>();
                }
            }
        }
        let shared = (mass[source] + mass[target]) / 2.0;
        mass[source] = shared;
        mass[target] = shared;
        usage[source] /= 2.0;
        usage[target] = usage[source];
    }
    n_split
}

pub struct FitResult {
    pub weights: Vec<f64>,
    pub means: Vec<f64>,
    pub covariances: Vec<f64>,
    pub lower_bound: f64,
    pub n_steps: usize,
    pub n_splits: usize,
}

/// The full e-GMM fit, from the (shifted) starting weights, means and variances.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    x: &[f64],
    n: usize,
    shift: &[f64],
    weights: &[f64],
    means: &[f64],
    variances: &[f64],
    dims: Dims,
    epoch_rows: &[usize],
    batch_size: usize,
    alphas: &[f64],
    reg_covar: f64,
    unused_threshold: f64,
    seed: u64,
) -> FitResult {
    let Dims { d, k, s, .. } = dims;
    let zd = dims.z();
    let mut rng = Rng::new(seed);
    let mut weights = weights.to_vec();
    let mut means = means.to_vec();
    let mut variances = variances.to_vec();
    let mut second = second_moments(&means, &variances, dims);
    let mut mass = vec![INIT_USAGE; k];
    let mut usage = vec![INIT_USAGE; k];
    let mut n_splits = 0usize;
    let mut order: Vec<usize> = (0..n).collect();
    let mut block = Block::default();
    let mut batch = Accum::new(dims);
    let mut pass = Accum::new(dims);
    let mut step = 0usize;

    for (epoch, &rows) in epoch_rows.iter().enumerate() {
        let last_pass = epoch + 1 == epoch_rows.len();
        rng.partial_shuffle(&mut order, rows);
        pass.reset();
        for start in (0..rows).step_by(batch_size.max(1)) {
            let index = &order[start..(start + batch_size).min(rows)];
            let (w, bias) = log_prob_params(&weights, &means, &variances, dims);
            block.estep(x, index, shift, &w, &bias, dims, &mut batch);
            if last_pass {
                pass.merge(&batch);
            }

            let alpha = alphas[step.min(alphas.len() - 1)];
            for (m, &bm) in mass.iter_mut().zip(&batch.mass) {
                *m = (1.0 - alpha) * *m + alpha * bm;
            }
            for (u, &c) in usage.iter_mut().zip(&batch.most_likely) {
                *u = (1.0 - alpha) * *u + alpha * c;
            }
            for j in 0..k {
                let bm = batch.mass[j];
                if bm <= MIN_MASS {
                    continue;
                }
                let step_size = alpha * bm / mass[j];
                let sums = &batch.sums[j * zd..(j + 1) * zd];
                for p in 0..d {
                    means[j * d + p] += step_size * (sums[p] / bm - means[j * d + p]);
                }
                for q in 0..s {
                    second[j * s + q] += step_size * (sums[d + q] / bm - second[j * s + q]);
                }
            }
            variances_from(&means, &second, reg_covar, dims, &mut variances);
            if (step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
                n_splits += revive_unused(
                    &mut means,
                    &mut second,
                    &mut variances,
                    &mut mass,
                    &mut usage,
                    dims,
                    unused_threshold,
                    &mut rng,
                );
            }
            let total: f64 = mass.iter().sum();
            for (w, &m) in weights.iter_mut().zip(&mass) {
                *w = (m / total).max(MIN_MASS);
            }
            let total: f64 = weights.iter().sum();
            for w in weights.iter_mut() {
                *w /= total;
            }
            step += 1;
        }
    }

    // A full M-step on the statistics of the last pass.
    for j in 0..k {
        let pm = pass.mass[j];
        if pm <= MIN_MASS {
            continue;
        }
        let sums = &pass.sums[j * zd..(j + 1) * zd];
        for p in 0..d {
            means[j * d + p] = sums[p] / pm;
        }
        for q in 0..s {
            second[j * s + q] = sums[d + q] / pm;
        }
    }
    let total: f64 = pass.mass.iter().map(|m| m + MIN_MASS).sum();
    let weights = pass.mass.iter().map(|m| (m + MIN_MASS) / total).collect();
    let mut covariances = vec![0.0; k * s];
    variances_from(&means, &second, reg_covar, dims, &mut covariances);
    let lower_bound = pass.log_likelihood / *epoch_rows.last().unwrap_or(&1) as f64;
    FitResult {
        weights,
        means,
        covariances,
        lower_bound,
        n_steps: step,
        n_splits,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// Log density of every row, shape (n,).
    ScoreSamples,
    /// Responsibilities, shape (n, k).
    Proba,
    /// Most likely component, shape (n,), returned as f64 and cast by the caller.
    Predict,
}

/// Evaluate the mixture on every row of `x`, in blocks of rows to bound memory.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    x: &[f64],
    n: usize,
    shift: &[f64],
    weights: &[f64],
    means: &[f64],
    variances: &[f64],
    dims: Dims,
    output: Output,
) -> Vec<f64> {
    let k = dims.k;
    let (w, bias) = log_prob_params(weights, means, variances, dims);
    let width = if output == Output::Proba { k } else { 1 };
    let mut out = vec![0.0; n * width];
    let mut block = Block::default();
    let rows_per_block = block_rows(k + dims.z());
    for (c, out) in out.chunks_mut(rows_per_block * width).enumerate() {
        let start = c * rows_per_block;
        let b = out.len() / width;
        block.log_prob(
            x,
            Rows::Range(start, start + b),
            shift,
            &w,
            &bias,
            dims,
            output != Output::Predict,
        );
        match output {
            Output::ScoreSamples => out.copy_from_slice(&block.log_norm),
            Output::Proba => out.copy_from_slice(&block.resp),
            Output::Predict => {
                for (o, row) in out.iter_mut().zip(block.resp.chunks_exact(k)) {
                    *o = argmax(row) as f64;
                }
            }
        }
    }
    out
}

/// Sums of the features `z` of the rows assigned to each label, shape (k, z), see `_initialize_from_emeans`.
pub fn label_stats(x: &[f64], n: usize, shift: &[f64], labels: &[i64], dims: Dims) -> Vec<f64> {
    let (d, k, zd) = (dims.d, dims.k, dims.z());
    let chunk = 4096;
    let parts: Vec<Vec<f64>> = (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|c| {
            let mut sums = vec![0.0; k * zd];
            let mut z = vec![0.0; zd];
            for i in c * chunk..((c + 1) * chunk).min(n) {
                features(&x[i * d..(i + 1) * d], shift, dims.cov, &mut z);
                let label = labels[i] as usize;
                for (s, &v) in sums[label * zd..(label + 1) * zd].iter_mut().zip(&z) {
                    *s += v;
                }
            }
            sums
        })
        .collect();
    let mut sums = vec![0.0; k * zd];
    for part in &parts {
        for (a, b) in sums.iter_mut().zip(part) {
            *a += b;
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revival_splits_a_busy_component_around_its_mean() {
        let dims = Dims::new(2, 3, Covariance::Diag);
        let mut means = vec![0.0, 0.0, 4.0, 4.0, 9.0, 9.0];
        let mut variances = vec![1.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        let mut second = second_moments(&means, &variances, dims);
        let mut mass = vec![5.0, 0.0, 5.0];
        let mut usage = vec![10.0, 0.0, 0.1];
        let n = revive_unused(
            &mut means,
            &mut second,
            &mut variances,
            &mut mass,
            &mut usage,
            dims,
            0.1,
            &mut Rng::new(0),
        );
        assert_eq!(n, 1);
        // Only component 0 is a donor (component 2 is below the limit as well, but there is one donor).
        let half = SPLIT_DISTANCE / 2f64.sqrt();
        assert_eq!(&means[0..2], &[-half * 1.0, half * 2.0]);
        assert_eq!(&means[2..4], &[half * 1.0, -half * 2.0]);
        assert_eq!(&variances[2..4], &[1.0, 4.0]);
        assert_eq!(second[2], 1.0 + half * half);
        assert_eq!((mass[0], mass[1]), (2.5, 2.5));
        assert_eq!((usage[0], usage[1]), (5.0, 5.0));
    }

    #[test]
    fn single_component_density_is_the_gaussian() {
        let dims = Dims::new(2, 1, Covariance::Diag);
        let x = vec![0.5, -1.0];
        let out = evaluate(
            &x,
            1,
            &[0.0, 0.0],
            &[1.0],
            &[0.0, 0.0],
            &[1.0, 4.0],
            dims,
            Output::ScoreSamples,
        );
        let expected = -LOG_2PI - 0.5 * (4.0f64).ln() - 0.5 * (0.25 + 1.0 / 4.0);
        assert!(
            (out[0] - expected).abs() < 1e-12,
            "{} vs {}",
            out[0],
            expected
        );
    }

    #[test]
    fn spherical_matches_diag_with_equal_variances() {
        let x = vec![0.3, 0.1, -0.2, 2.0, 1.5, 0.7];
        let means = vec![0.0, 0.0, 0.0, 2.0, 2.0, 1.0];
        let weights = [0.4, 0.6];
        let diag = evaluate(
            &x,
            2,
            &[0.0; 3],
            &weights,
            &means,
            &[0.5, 0.5, 0.5, 2.0, 2.0, 2.0],
            Dims::new(3, 2, Covariance::Diag),
            Output::Proba,
        );
        let sph = evaluate(
            &x,
            2,
            &[0.0; 3],
            &weights,
            &means,
            &[0.5, 2.0],
            Dims::new(3, 2, Covariance::Spherical),
            Output::Proba,
        );
        for (a, b) in diag.iter().zip(&sph) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
