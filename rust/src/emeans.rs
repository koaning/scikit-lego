//! e-means: a port of the training loop of `sklego.cluster.EMeans`.
//!
//! Python resolves every default (batch size, step sizes, revival threshold, epoch lengths) and the starting
//! centroids. This module runs all epochs: the shuffling, the mini-batch updates, the revivals, the exact final
//! means and the final assignment of all rows.

use rayon::prelude::*;

use crate::linalg::{matmul_bt, squared_norm, Float, Rng};

const CHECK_UNUSED_EVERY: usize = 5;
const INIT_USAGE: f64 = 1.0 / 65536.0;
const SPLIT_EPSILON: f64 = 1e-4;

/// Elements of the (rows, clusters) matrix kept in memory at once in full-data passes, like `_CHUNK_ELEMENTS`.
const BLOCK_ELEMENTS: usize = 1 << 22;
/// Blocks with fewer (rows * columns) elements than this are post-processed on one thread: waking up the thread
/// pool costs more than it saves on smaller blocks.
pub const PARALLEL_MIN: usize = 1 << 18;
/// Rows per rayon task when post-processing a block.
pub const MIN_ROWS_PER_TASK: usize = 64;

/// Rows per block so that a (rows, n_cols) matrix holds about `BLOCK_ELEMENTS` values.
pub fn block_rows(n_cols: usize) -> usize {
    (BLOCK_ELEMENTS / n_cols.max(1)).max(1)
}

/// Per-cluster row counts, float64 row sums and the sum of squared distances of a mini-batch or a pass.
pub struct Accum {
    pub counts: Vec<f64>,
    pub sums: Vec<f64>,
    pub objective: f64,
}

impl Accum {
    pub fn new(k: usize, d: usize) -> Self {
        Accum {
            counts: vec![0.0; k],
            sums: vec![0.0; k * d],
            objective: 0.0,
        }
    }

    fn reset(&mut self) {
        self.counts.fill(0.0);
        self.sums.fill(0.0);
        self.objective = 0.0;
    }

    fn merge(&mut self, other: &Accum) {
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
        for (a, b) in self.sums.iter_mut().zip(&other.sums) {
            *a += b;
        }
        self.objective += other.objective;
    }
}

/// Nearest centroid of a row given its row of `X @ centers.T`. Row norms do not change the argmin, so they are
/// only added to the winning distance, as in the NumPy backend. Ties go to the lowest index, like `np.argmin`.
#[inline]
fn argmin_row<T: Float>(gram_row: &[T], center_norms: &[T], row_norm: T) -> (usize, T) {
    // Independent lanes break the dependency between comparisons, so the loop vectorises.
    const LANES: usize = 8;
    let k = gram_row.len();
    let (gram_chunks, _) = gram_row.as_chunks::<LANES>();
    let (norm_chunks, _) = center_norms.as_chunks::<LANES>();
    let full = gram_chunks.len() * LANES;
    let (mut best, mut best_value) = (0, T::MINUS_TWO * gram_row[0] + center_norms[0]);
    if full > 0 {
        let mut lane_value = [best_value; LANES];
        let mut lane_index = [0usize; LANES];
        for (c, (g, cn)) in gram_chunks.iter().zip(norm_chunks).enumerate() {
            for l in 0..LANES {
                let value = T::MINUS_TWO * g[l] + cn[l];
                if value < lane_value[l] {
                    lane_value[l] = value;
                    lane_index[l] = c * LANES + l;
                }
            }
        }
        for l in 0..LANES {
            if lane_value[l] < best_value || (lane_value[l] == best_value && lane_index[l] < best) {
                best_value = lane_value[l];
                best = lane_index[l];
            }
        }
    }
    for j in full..k {
        let value = T::MINUS_TWO * gram_row[j] + center_norms[j];
        if value < best_value {
            best = j;
            best_value = value;
        }
    }
    let dist = best_value + row_norm;
    (best, if dist > T::ZERO { dist } else { T::ZERO })
}

/// Nearest centroid and squared distance of the contiguous rows `rows`: one GEMM, then the argmin of every row on
/// several threads when the block is large. `gram` is scratch space.
#[allow(clippy::too_many_arguments)]
fn assign_block<T: Float>(
    rows: &[T],
    d: usize,
    centers: &[T],
    center_norms: &[T],
    k: usize,
    gram: &mut Vec<T>,
    labels: &mut [usize],
    dist: &mut [T],
) {
    let b = labels.len();
    gram.resize(b * k, T::ZERO);
    matmul_bt(rows, centers, b, d, k, gram);
    let assign = |(r, (label, dist)): (usize, (&mut usize, &mut T))| {
        (*label, *dist) = argmin_row(
            &gram[r * k..(r + 1) * k],
            center_norms,
            squared_norm(&rows[r * d..(r + 1) * d]),
        );
    };
    if b * k < PARALLEL_MIN {
        labels
            .iter_mut()
            .zip(dist.iter_mut())
            .enumerate()
            .for_each(assign);
    } else {
        labels
            .par_iter_mut()
            .zip(dist.par_iter_mut())
            .enumerate()
            .with_min_len(MIN_ROWS_PER_TASK)
            .for_each(assign);
    }
}

/// Buffers reused across mini-batches: gathered rows, `rows @ centers.T`, labels and distances.
struct Batch<T> {
    rows: Vec<T>,
    gram: Vec<T>,
    labels: Vec<usize>,
    dist: Vec<T>,
}

impl<T: Float> Batch<T> {
    fn new() -> Self {
        Batch {
            rows: Vec::new(),
            gram: Vec::new(),
            labels: Vec::new(),
            dist: Vec::new(),
        }
    }

    /// Assign the rows `index` of `x` and put their counts, sums and distances in `acc`.
    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &mut self,
        x: &[T],
        d: usize,
        index: &[usize],
        centers: &[T],
        norms: &[T],
        k: usize,
        acc: &mut Accum,
    ) {
        let b = index.len();
        self.rows.resize(b * d, T::ZERO);
        let gather = |(dst, &i): (&mut [T], &usize)| dst.copy_from_slice(&x[i * d..(i + 1) * d]);
        if d == 0 {
        } else if b * d < PARALLEL_MIN {
            self.rows.chunks_exact_mut(d).zip(index).for_each(gather);
        } else {
            self.rows
                .par_chunks_exact_mut(d)
                .zip(index.par_iter())
                .with_min_len(MIN_ROWS_PER_TASK)
                .for_each(gather);
        }
        self.labels.resize(b, 0);
        self.dist.resize(b, T::ZERO);
        assign_block(
            &self.rows,
            d,
            centers,
            norms,
            k,
            &mut self.gram,
            &mut self.labels,
            &mut self.dist,
        );

        acc.reset();
        for r in 0..b {
            let label = self.labels[r];
            acc.objective += self.dist[r].to_f64();
            acc.counts[label] += 1.0;
            let sums = &mut acc.sums[label * d..(label + 1) * d];
            for (s, &v) in sums.iter_mut().zip(&self.rows[r * d..(r + 1) * d]) {
                *s += v.to_f64();
            }
        }
    }
}

/// Moving average update of the centroids and their masses, see `EMeans._apply_batch`.
fn apply_batch<T: Float>(centers: &mut [T], usage: &mut [f64], d: usize, acc: &Accum, alpha: f64) {
    if alpha < 1.0 {
        for u in usage.iter_mut() {
            *u *= 1.0 - alpha;
        }
    }
    for (j, &count) in acc.counts.iter().enumerate() {
        if count <= 0.0 {
            continue;
        }
        let retained = if alpha < 1.0 { usage[j] } else { 0.0 };
        let mass = retained + alpha * count;
        let weight = alpha * count / mass;
        let center = &mut centers[j * d..(j + 1) * d];
        for (c, &s) in center.iter_mut().zip(&acc.sums[j * d..(j + 1) * d]) {
            let value = c.to_f64();
            *c = T::from_f64(value + weight * (s / count - value));
        }
        usage[j] = mass;
    }
}

/// Split busy clusters, drawn in proportion to their mass, to replace the ones with almost no mass.
/// See `EMeans._revive_unused`.
pub fn revive_unused<T: Float>(
    centers: &mut [T],
    usage: &mut [f64],
    d: usize,
    threshold: f64,
    rng: &mut Rng,
) -> usize {
    if threshold == 0.0 {
        return 0;
    }
    let k = usage.len();
    let limit = usage.iter().sum::<f64>() * threshold / k as f64;
    let unused: Vec<usize> = (0..k).filter(|&j| usage[j] < limit).collect();
    let donors: Vec<usize> = (0..k)
        .filter(|&j| usage[j] >= limit && usage[j] > 0.0)
        .collect();
    let n_split = unused.len().min(donors.len());
    if n_split == 0 {
        return 0;
    }
    // Gumbel top-k: donors drawn without replacement in proportion to their mass.
    let mut keyed: Vec<(f64, usize)> = donors
        .iter()
        .map(|&j| (usage[j].ln() - (-rng.uniform().max(1e-300).ln()).ln(), j))
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (t, &(_, source)) in keyed.iter().take(n_split).enumerate() {
        let target = unused[t];
        let mut collapsed = true;
        for p in 0..d {
            let offset = if p % 2 == 0 {
                SPLIT_EPSILON
            } else {
                -SPLIT_EPSILON
            };
            let base = centers[source * d + p].to_f64();
            let a = T::from_f64(base * (1.0 - offset));
            let b = T::from_f64(base * (1.0 + offset));
            centers[source * d + p] = a;
            centers[target * d + p] = b;
            collapsed &= a == b;
        }
        if collapsed && d > 0 {
            centers[target * d] = centers[target * d].next_up();
        }
        usage[source] /= 2.0;
        usage[target] = usage[source];
    }
    n_split
}

fn center_norms<T: Float>(centers: &[T], d: usize, out: &mut [T]) {
    for (norm, row) in out.iter_mut().zip(centers.chunks_exact(d.max(1))) {
        *norm = squared_norm(&row[..d]);
    }
}

/// Nearest centroid and squared distance of every row of `x`, in blocks of rows to bound memory.
pub fn nearest<T: Float>(
    x: &[T],
    n: usize,
    d: usize,
    centers: &[T],
    k: usize,
) -> (Vec<i64>, Vec<T>) {
    let mut norms = vec![T::ZERO; k];
    center_norms(centers, d, &mut norms);
    let mut labels = vec![0usize; n];
    let mut dist = vec![T::ZERO; n];
    let mut gram = Vec::new();
    let block = block_rows(k);
    for (c, (labels, dist)) in labels
        .chunks_mut(block)
        .zip(dist.chunks_mut(block))
        .enumerate()
    {
        let rows = &x[c * block * d..(c * block + labels.len()) * d];
        assign_block(rows, d, centers, &norms, k, &mut gram, labels, dist);
    }
    (labels.into_iter().map(|l| l as i64).collect(), dist)
}

/// Euclidean distance of every row of `x` to every centroid, shape (n, k).
pub fn distances<T: Float>(x: &[T], n: usize, d: usize, centers: &[T], k: usize) -> Vec<T> {
    let mut norms = vec![T::ZERO; k];
    center_norms(centers, d, &mut norms);
    let mut out = vec![T::ZERO; n * k];
    if k == 0 {
        return out;
    }
    let block = block_rows(k);
    for (c, out) in out.chunks_mut(block * k).enumerate() {
        let b = out.len() / k;
        let rows = &x[c * block * d..(c * block + b) * d];
        matmul_bt(rows, centers, b, d, k, out);
        let finish = |(r, row): (usize, &mut [T])| {
            let row_norm = squared_norm(&rows[r * d..(r + 1) * d]);
            for (v, &cn) in row.iter_mut().zip(&norms) {
                let squared = T::MINUS_TWO * *v + cn + row_norm;
                *v = if squared > T::ZERO {
                    squared.sqrt()
                } else {
                    T::ZERO
                };
            }
        };
        if b * k < PARALLEL_MIN {
            out.chunks_exact_mut(k).enumerate().for_each(finish);
        } else {
            out.par_chunks_exact_mut(k)
                .enumerate()
                .with_min_len(MIN_ROWS_PER_TASK)
                .for_each(finish);
        }
    }
    out
}

pub struct FitResult<T> {
    pub centers: Vec<T>,
    pub labels: Vec<i64>,
    pub inertia: f64,
    pub n_steps: usize,
    pub n_splits: usize,
    pub objective_history: Vec<f64>,
}

/// The full e-means fit. `alphas[step]` is the step size of every mini-batch update.
#[allow(clippy::too_many_arguments)]
pub fn fit<T: Float>(
    x: &[T],
    n: usize,
    d: usize,
    init: &[T],
    k: usize,
    epoch_rows: &[usize],
    batch_size: usize,
    alphas: &[f64],
    threshold: f64,
    seed: u64,
) -> FitResult<T> {
    let mut rng = Rng::new(seed);
    let mut centers = init.to_vec();
    let mut usage = vec![INIT_USAGE; k];
    let mut norms = vec![T::ZERO; k];
    let mut order: Vec<usize> = (0..n).collect();
    center_norms(&centers, d, &mut norms);
    let mut buffers = Batch::new();
    let mut batch = Accum::new(k, d);
    let mut pass = Accum::new(k, d);
    let (mut step, mut n_splits) = (0usize, 0usize);
    let mut objective_history = Vec::with_capacity(epoch_rows.len());

    for (epoch, &rows) in epoch_rows.iter().enumerate() {
        let last_pass = epoch + 1 == epoch_rows.len();
        rng.partial_shuffle(&mut order, rows);
        pass.reset();
        let mut objective = 0.0;
        for start in (0..rows).step_by(batch_size.max(1)) {
            let index = &order[start..(start + batch_size).min(rows)];
            buffers.accumulate(x, d, index, &centers, &norms, k, &mut batch);
            objective += batch.objective;
            if last_pass {
                pass.merge(&batch);
            }
            apply_batch(
                &mut centers,
                &mut usage,
                d,
                &batch,
                alphas[step.min(alphas.len() - 1)],
            );
            // Only the centroids that received rows moved, unless some were revived.
            let mut revived = 0;
            if (step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
                revived = revive_unused(&mut centers, &mut usage, d, threshold, &mut rng);
                n_splits += revived;
            }
            if revived > 0 {
                center_norms(&centers, d, &mut norms);
            } else {
                for (j, norm) in norms.iter_mut().enumerate() {
                    if batch.counts[j] > 0.0 {
                        *norm = squared_norm(&centers[j * d..(j + 1) * d]);
                    }
                }
            }
            step += 1;
        }
        objective_history.push(objective);
    }

    // Every centroid becomes the exact mean of the rows it received during the last pass.
    for j in 0..k {
        let count = pass.counts[j];
        if count > 0.0 {
            for p in 0..d {
                centers[j * d + p] = T::from_f64(pass.sums[j * d + p] / count);
            }
        }
    }

    let (labels, dist) = nearest(x, n, d, &centers, k);
    let inertia = dist.iter().map(|v| v.to_f64()).sum();
    FitResult {
        centers,
        labels,
        inertia,
        n_steps: step,
        n_splits,
        objective_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revival_splits_the_heaviest_donors() {
        let d = 2;
        let mut centers = vec![1.0f64, 1.0, 2.0, 2.0, 3.0, 3.0];
        let mut usage = vec![10.0, 0.0, 5.0];
        let mut rng = Rng::new(0);
        let n = revive_unused(&mut centers, &mut usage, d, 0.1, &mut rng);
        assert_eq!(n, 1);
        // Cluster 1 is revived as a copy of one of the donors, which keeps half of its mass.
        assert!(
            usage == vec![5.0, 5.0, 5.0] || usage == vec![10.0, 2.5, 2.5],
            "{usage:?}"
        );
        assert!(centers[2] != 2.0);
    }

    #[test]
    fn argmin_matches_a_plain_scan_and_breaks_ties_low() {
        let mut rng = Rng::new(5);
        for k in [1usize, 3, 8, 9, 17, 64, 100] {
            for _ in 0..50 {
                // Coarse values so that ties happen.
                let gram: Vec<f64> = (0..k).map(|_| (rng.below(4) as f64) * 0.5).collect();
                let norms: Vec<f64> = (0..k).map(|_| rng.below(3) as f64).collect();
                let values: Vec<f64> = gram.iter().zip(&norms).map(|(g, c)| -2.0 * g + c).collect();
                let mut expected = 0;
                for j in 1..k {
                    if values[j] < values[expected] {
                        expected = j;
                    }
                }
                let (label, _) = argmin_row(&gram, &norms, 10.0);
                assert_eq!(label, expected, "k={k} {values:?}");
            }
        }
    }

    #[test]
    fn no_revival_when_threshold_is_zero() {
        let mut centers = vec![0.0f32; 4];
        let mut usage = vec![1.0, 0.0];
        assert_eq!(
            revive_unused(&mut centers, &mut usage, 2, 0.0, &mut Rng::new(0)),
            0
        );
    }

    #[test]
    fn revived_centroid_differs_from_a_zero_donor() {
        let mut centers = vec![0.0f64; 4];
        let mut usage = vec![1.0, 0.0];
        revive_unused(&mut centers, &mut usage, 2, 0.5, &mut Rng::new(0));
        assert_ne!(centers[0..2], centers[2..4]);
        assert_eq!(usage, vec![0.5, 0.5]);
    }

    #[test]
    fn fit_recovers_two_clusters() {
        let mut x = Vec::new();
        let mut rng = Rng::new(42);
        for i in 0..400 {
            let offset = if i % 2 == 0 { -5.0 } else { 5.0 };
            x.push(offset + rng.uniform() - 0.5);
            x.push(rng.uniform() - 0.5);
        }
        let init = vec![x[0], x[1], x[2], x[3]];
        let alphas = vec![0.5; 100];
        let result = fit(&x, 400, 2, &init, 2, &[400, 400], 32, &alphas, 0.0, 0);
        let mut first: Vec<f64> = result.centers.chunks(2).map(|c| c[0]).collect();
        first.sort_by(f64::total_cmp);
        assert!(
            (first[0] + 5.0).abs() < 0.2 && (first[1] - 5.0).abs() < 0.2,
            "{first:?}"
        );
        assert_eq!(result.n_steps, 26);
        assert_eq!(result.objective_history.len(), 2);
    }
}
