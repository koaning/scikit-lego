//! Numeric helpers: a float trait over f32/f64, a row-major GEMM and a seeded RNG.
//!
//! GEMM goes through the BLAS that scipy ships (`scipy.linalg.cython_blas`), the same approach scikit-learn's
//! `KMeans` uses. That is Accelerate on recent macOS wheels and OpenBLAS elsewhere, so the matrix products are as
//! fast as NumPy's. When scipy's BLAS cannot be loaded (or `use_blas(false)` is set), the pure Rust
//! `matrixmultiply` crate is used instead.

use std::ffi::{c_char, c_int};
use std::ops::{Add, AddAssign, Mul, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Fortran `?gemm` as exported by `scipy.linalg.cython_blas`.
type FortranGemm<T> = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const T,
    a: *const T,
    lda: *const c_int,
    b: *const T,
    ldb: *const c_int,
    beta: *const T,
    c: *mut T,
    ldc: *const c_int,
);

pub struct Blas {
    sgemm: FortranGemm<f32>,
    dgemm: FortranGemm<f64>,
}

static BLAS: OnceLock<Option<Blas>> = OnceLock::new();
static USE_BLAS: AtomicBool = AtomicBool::new(true);

/// Register scipy's BLAS. Called once when the Python module is imported.
///
/// # Safety
/// The pointers must be scipy's Fortran `sgemm` and `dgemm`.
pub unsafe fn set_blas(sgemm: *mut std::ffi::c_void, dgemm: *mut std::ffi::c_void) {
    let _ = BLAS.set(Some(Blas {
        sgemm: std::mem::transmute::<*mut std::ffi::c_void, FortranGemm<f32>>(sgemm),
        dgemm: std::mem::transmute::<*mut std::ffi::c_void, FortranGemm<f64>>(dgemm),
    }));
}

/// Turn the use of scipy's BLAS on or off (it is on when it could be loaded).
pub fn use_blas(on: bool) {
    USE_BLAS.store(on, Ordering::Relaxed);
}

/// Whether matrix products currently go through scipy's BLAS.
pub fn blas_active() -> bool {
    USE_BLAS.load(Ordering::Relaxed) && matches!(BLAS.get(), Some(Some(_)))
}

fn blas() -> Option<&'static Blas> {
    if !USE_BLAS.load(Ordering::Relaxed) {
        return None;
    }
    BLAS.get().and_then(|b| b.as_ref())
}

/// The float types the kernels work in (float32 or float64, like the NumPy backend).
pub trait Float:
    Copy
    + Send
    + Sync
    + PartialOrd
    + Default
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const MINUS_TWO: Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    /// The next representable value towards +inf, like `np.nextafter(x, np.inf)`.
    fn next_up(self) -> Self;
    fn fortran_gemm(blas: &Blas) -> FortranGemm<Self>;
    /// `matrixmultiply` fallback: `c = alpha * a @ b + beta * c` with explicit (row, column) strides.
    ///
    /// # Safety
    /// The pointers and strides must describe valid (m x k), (k x n) and (m x n) matrices.
    #[allow(clippy::too_many_arguments)]
    unsafe fn rust_gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    );
}

macro_rules! impl_float {
    ($t:ty, $bits:ty, $rust_gemm:path, $field:ident) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const MINUS_TWO: Self = -2.0;
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn next_up(self) -> Self {
                if self.is_nan() || self == <$t>::INFINITY {
                    return self;
                }
                if self == 0.0 {
                    return <$t>::from_bits(1);
                }
                let bits = self.to_bits();
                <$t>::from_bits(if self > 0.0 { bits + 1 } else { bits - 1 })
            }
            fn fortran_gemm(blas: &Blas) -> FortranGemm<Self> {
                blas.$field
            }
            unsafe fn rust_gemm(
                m: usize,
                k: usize,
                n: usize,
                alpha: Self,
                a: *const Self,
                rsa: isize,
                csa: isize,
                b: *const Self,
                rsb: isize,
                csb: isize,
                beta: Self,
                c: *mut Self,
                rsc: isize,
                csc: isize,
            ) {
                $rust_gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
            }
        }
    };
}

impl_float!(f32, u32, matrixmultiply::sgemm, sgemm);
impl_float!(f64, u64, matrixmultiply::dgemm, dgemm);

/// Row-major `c (m x n) = op(a) @ op(b) + beta * c`, where `op(a)` is (m x kk) and `op(b)` is (kk x n).
///
/// `a` is stored (m x kk) with row stride `lda`, or (kk x m) when `trans_a`. The same goes for `b`. `c` has row
/// stride `n`.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: Float>(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    kk: usize,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
) {
    if m == 0 || n == 0 {
        return;
    }
    debug_assert!(c.len() >= m * n);
    if kk == 0 {
        for v in c[..m * n].iter_mut() {
            *v = beta * *v;
        }
        return;
    }
    let fits = |v: usize| v <= c_int::MAX as usize;
    if let Some(blas) = blas().filter(|_| fits(m) && fits(n) && fits(kk) && fits(lda) && fits(ldb))
    {
        // Column-major BLAS sees every row-major matrix transposed: c.T = op(b).T @ op(a).T.
        let (ta, tb) = (
            if trans_a { b'T' } else { b'N' },
            if trans_b { b'T' } else { b'N' },
        );
        let (mi, ni, ki) = (n as c_int, m as c_int, kk as c_int);
        let (lda_i, ldb_i, ldc_i) = (lda as c_int, ldb as c_int, n as c_int);
        let one = T::ONE;
        unsafe {
            (T::fortran_gemm(blas))(
                &tb as *const u8 as *const c_char,
                &ta as *const u8 as *const c_char,
                &mi,
                &ni,
                &ki,
                &one,
                b.as_ptr(),
                &ldb_i,
                a.as_ptr(),
                &lda_i,
                &beta,
                c.as_mut_ptr(),
                &ldc_i,
            );
        }
        return;
    }
    let (rsa, csa) = if trans_a {
        (1, lda as isize)
    } else {
        (lda as isize, 1)
    };
    let (rsb, csb) = if trans_b {
        (1, ldb as isize)
    } else {
        (ldb as isize, 1)
    };
    unsafe {
        T::rust_gemm(
            m,
            kk,
            n,
            T::ONE,
            a.as_ptr(),
            rsa,
            csa,
            b.as_ptr(),
            rsb,
            csb,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1,
        )
    }
}

/// `out (m x n) = a (m x k) @ b.T`, with `a` and `b` (n x k) row-major and contiguous.
#[inline]
pub fn matmul_bt<T: Float>(a: &[T], b: &[T], m: usize, k: usize, n: usize, out: &mut [T]) {
    gemm(false, true, m, n, k, a, k, b, k, T::ZERO, out);
}

/// `out (m x n) = a (m x k) @ b (k x n)`, all row-major and contiguous.
#[inline]
pub fn matmul<T: Float>(a: &[T], b: &[T], m: usize, k: usize, n: usize, out: &mut [T]) {
    gemm(false, false, m, n, k, a, k, b, n, T::ZERO, out);
}

/// `out (k x n) += a.T @ b` with `a` (m x k) and `b` (m x n), all row-major and contiguous.
#[inline]
pub fn add_matmul_at<T: Float>(a: &[T], b: &[T], m: usize, k: usize, n: usize, out: &mut [T]) {
    gemm(true, false, k, n, m, a, k, b, n, T::ONE, out);
}

#[inline]
pub fn squared_norm<T: Float>(row: &[T]) -> T {
    let mut s = T::ZERO;
    for &v in row {
        s += v * v;
    }
    s
}

/// xoshiro256++ seeded through splitmix64: small, fast and good enough for shuffling and Gumbel keys.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, bound)`, without modulo bias (Lemire's method).
    #[inline]
    pub fn below(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let mut m = (self.next_u64() as u128) * (bound as u128);
        if (m as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (m as u64) < threshold {
                m = (self.next_u64() as u128) * (bound as u128);
            }
        }
        (m >> 64) as usize
    }

    /// Put a uniformly random selection of `take` elements, in random order, at the front of `values`.
    pub fn partial_shuffle(&mut self, values: &mut [usize], take: usize) {
        let n = values.len();
        for i in 0..take.min(n.saturating_sub(1)) {
            let j = i + self.below(n - i);
            values.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(7);
        let mut v: Vec<usize> = (0..1000).collect();
        rng.partial_shuffle(&mut v, 1000);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
        assert_ne!(v, sorted);
    }

    #[test]
    fn below_is_roughly_uniform() {
        let mut rng = Rng::new(1);
        let mut counts = [0usize; 10];
        for _ in 0..100_000 {
            counts[rng.below(10)] += 1;
        }
        for c in counts {
            assert!((9_000..11_000).contains(&c), "{counts:?}");
        }
    }

    #[test]
    fn uniform_in_unit_interval() {
        let mut rng = Rng::new(3);
        let mean: f64 = (0..100_000).map(|_| rng.uniform()).sum::<f64>() / 100_000.0;
        assert!((mean - 0.5).abs() < 0.01);
    }

    fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, ta: bool, tb: bool) -> Vec<f64> {
        let at = |i: usize, p: usize| if ta { a[p * m + i] } else { a[i * k + p] };
        let bt = |p: usize, j: usize| if tb { b[j * k + p] } else { b[p * n + j] };
        (0..m * n)
            .map(|ij| (0..k).map(|p| at(ij / n, p) * bt(p, ij % n)).sum())
            .collect()
    }

    #[test]
    fn gemm_variants_match_naive() {
        let (m, k, n) = (3, 4, 5);
        let a: Vec<f64> = (0..m * k).map(|v| v as f64 * 0.5 - 2.0).collect();
        let b: Vec<f64> = (0..k * n).map(|v| (v as f64).sin()).collect();
        for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut out = vec![0.0; m * n];
            let lda = if ta { m } else { k };
            let ldb = if tb { k } else { n };
            gemm(ta, tb, m, n, k, &a, lda, &b, ldb, 0.0, &mut out);
            let expected = naive(&a, &b, m, k, n, ta, tb);
            for (x, y) in out.iter().zip(&expected) {
                assert!((x - y).abs() < 1e-12, "{ta} {tb}");
            }
        }
    }

    #[test]
    fn next_up_moves_one_ulp() {
        assert!(1.0f64.next_up() > 1.0);
        assert!(1.0f32.next_up() > 1.0);
        assert_eq!((-0.0f64).next_up(), f64::from_bits(1));
    }
}
