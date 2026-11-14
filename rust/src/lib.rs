//! Python bindings of the e-means and e-GMM kernels used by `sklego.cluster.EMeans(backend="rust")` and
//! `sklego.mixture.EGaussianMixture(backend="rust")`.
//!
//! These functions are internal to scikit-lego: they expect C-contiguous float arrays and parameters that the Python
//! estimators already validated and resolved. The k-means functions come in a `_float32` and a `_float64` flavour.

pub mod egmm;
pub mod emeans;
pub mod linalg;

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use egmm::{Covariance, Dims, Output};

fn slice<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    array: &'a numpy::PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn check_shape(name: &str, actual: &[usize], expected: &[usize]) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "{name} has shape {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

type EMeansOutput<'py, T> = (
    Bound<'py, PyArray2<T>>,
    Bound<'py, PyArray1<i64>>,
    f64,
    usize,
    usize,
    Vec<f64>,
);

macro_rules! emeans_functions {
    ($t:ty, $fit:ident, $nearest:ident, $distances:ident) => {
        /// Fit e-means from `centers`. Returns (centers, labels, inertia, n_steps, n_splits, objective_history).
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $fit<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, $t>,
            centers: PyReadonlyArray2<'py, $t>,
            epoch_rows: Vec<usize>,
            batch_size: usize,
            alphas: PyReadonlyArray1<'py, f64>,
            unused_threshold: f64,
            seed: u64,
        ) -> PyResult<EMeansOutput<'py, $t>> {
            let (n, d) = (x.shape()[0], x.shape()[1]);
            let k = centers.shape()[0];
            check_shape("centers", centers.shape(), &[k, d])?;
            let total_steps: usize = epoch_rows
                .iter()
                .map(|r| r.div_ceil(batch_size.max(1)))
                .sum();
            if batch_size == 0
                || k == 0
                || alphas.len() < total_steps
                || epoch_rows.iter().any(|&r| r > n)
            {
                return Err(PyValueError::new_err("inconsistent e-means parameters"));
            }
            let (xs, cs, al) = (
                slice(&x, "X")?,
                slice(&centers, "centers")?,
                slice(&alphas, "alphas")?,
            );
            let result = py.detach(|| {
                emeans::fit(
                    xs,
                    n,
                    d,
                    cs,
                    k,
                    &epoch_rows,
                    batch_size,
                    al,
                    unused_threshold,
                    seed,
                )
            });
            Ok((
                PyArray1::from_vec(py, result.centers).reshape([k, d])?,
                PyArray1::from_vec(py, result.labels),
                result.inertia,
                result.n_steps,
                result.n_splits,
                result.objective_history,
            ))
        }

        /// Nearest centroid and squared distance of every row.
        #[pyfunction]
        fn $nearest<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, $t>,
            centers: PyReadonlyArray2<'py, $t>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$t>>)> {
            let (n, d) = (x.shape()[0], x.shape()[1]);
            let k = centers.shape()[0];
            check_shape("centers", centers.shape(), &[k, d])?;
            let (xs, cs) = (slice(&x, "X")?, slice(&centers, "centers")?);
            let (labels, dist) = py.detach(|| emeans::nearest(xs, n, d, cs, k));
            Ok((PyArray1::from_vec(py, labels), PyArray1::from_vec(py, dist)))
        }

        /// Euclidean distance of every row to every centroid.
        #[pyfunction]
        fn $distances<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'py, $t>,
            centers: PyReadonlyArray2<'py, $t>,
        ) -> PyResult<Bound<'py, PyArray2<$t>>> {
            let (n, d) = (x.shape()[0], x.shape()[1]);
            let k = centers.shape()[0];
            check_shape("centers", centers.shape(), &[k, d])?;
            let (xs, cs) = (slice(&x, "X")?, slice(&centers, "centers")?);
            let out = py.detach(|| emeans::distances(xs, n, d, cs, k));
            PyArray1::from_vec(py, out).reshape([n, k])
        }
    };
}

emeans_functions!(
    f32,
    emeans_fit_float32,
    emeans_nearest_float32,
    emeans_distances_float32
);
emeans_functions!(
    f64,
    emeans_fit_float64,
    emeans_nearest_float64,
    emeans_distances_float64
);

fn covariance(name: &str) -> PyResult<Covariance> {
    Covariance::parse(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "covariance_type must be 'diag' or 'spherical', got {name}"
        ))
    })
}

/// Check the shapes of a mixture and return its dimensions.
fn mixture_dims(
    d: usize,
    shift: &PyReadonlyArray1<f64>,
    weights: &PyReadonlyArray1<f64>,
    means: &PyReadonlyArray2<f64>,
    variances: &numpy::PyReadonlyArrayDyn<f64>,
    covariance_type: &str,
) -> PyResult<Dims> {
    let k = weights.len();
    let dims = Dims::new(d, k, covariance(covariance_type)?);
    check_shape("shift", shift.shape(), &[d])?;
    check_shape("means", means.shape(), &[k, d])?;
    match dims.cov {
        Covariance::Diag => check_shape("variances", variances.shape(), &[k, d])?,
        Covariance::Spherical => check_shape("variances", variances.shape(), &[k])?,
    }
    if k == 0 {
        return Err(PyValueError::new_err(
            "the mixture needs at least one component",
        ));
    }
    Ok(dims)
}

type EgmmOutput<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyAny>,
    f64,
    usize,
    usize,
);

/// Fit e-GMM on `x - shift`. Returns (weights, means, covariances, lower_bound, n_steps, n_splits), means still
/// shifted.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn egmm_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    shift: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    means: PyReadonlyArray2<'py, f64>,
    variances: numpy::PyReadonlyArrayDyn<'py, f64>,
    covariance_type: &str,
    epoch_rows: Vec<usize>,
    batch_size: usize,
    alphas: PyReadonlyArray1<'py, f64>,
    reg_covar: f64,
    unused_threshold: f64,
    seed: u64,
) -> PyResult<EgmmOutput<'py>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let dims = mixture_dims(d, &shift, &weights, &means, &variances, covariance_type)?;
    let total_steps: usize = epoch_rows
        .iter()
        .map(|r| r.div_ceil(batch_size.max(1)))
        .sum();
    if batch_size == 0
        || epoch_rows.is_empty()
        || alphas.len() < total_steps
        || epoch_rows.iter().any(|&r| r > n)
    {
        return Err(PyValueError::new_err("inconsistent e-GMM parameters"));
    }
    let xs = slice(&x, "X")?;
    let (sh, w, m, v, al) = (
        slice(&shift, "shift")?,
        slice(&weights, "weights")?,
        slice(&means, "means")?,
        slice(&variances, "variances")?,
        slice(&alphas, "alphas")?,
    );
    let result = py.detach(|| {
        egmm::fit(
            xs,
            n,
            sh,
            w,
            m,
            v,
            dims,
            &epoch_rows,
            batch_size,
            al,
            reg_covar,
            unused_threshold,
            seed,
        )
    });
    let covariances = PyArray1::from_vec(py, result.covariances);
    let covariances = match dims.cov {
        Covariance::Diag => covariances.reshape([dims.k, d])?.into_any(),
        Covariance::Spherical => covariances.into_any(),
    };
    Ok((
        PyArray1::from_vec(py, result.weights),
        PyArray1::from_vec(py, result.means).reshape([dims.k, d])?,
        covariances,
        result.lower_bound,
        result.n_steps,
        result.n_splits,
    ))
}

/// Evaluate a mixture on `x`: `output` is "score_samples", "predict_proba" or "predict".
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn egmm_evaluate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    shift: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    means: PyReadonlyArray2<'py, f64>,
    variances: numpy::PyReadonlyArrayDyn<'py, f64>,
    covariance_type: &str,
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let dims = mixture_dims(d, &shift, &weights, &means, &variances, covariance_type)?;
    let mode = match output {
        "score_samples" => Output::ScoreSamples,
        "predict_proba" => Output::Proba,
        "predict" => Output::Predict,
        _ => return Err(PyValueError::new_err(format!("unknown output {output}"))),
    };
    let xs = slice(&x, "X")?;
    let (sh, w, m, v) = (
        slice(&shift, "shift")?,
        slice(&weights, "weights")?,
        slice(&means, "means")?,
        slice(&variances, "variances")?,
    );
    let out = py.detach(|| egmm::evaluate(xs, n, sh, w, m, v, dims, mode));
    Ok(match mode {
        Output::ScoreSamples => PyArray1::from_vec(py, out).into_any(),
        Output::Proba => PyArray1::from_vec(py, out).reshape([n, dims.k])?.into_any(),
        Output::Predict => {
            PyArray1::from_vec(py, out.into_iter().map(|v| v as i64).collect()).into_any()
        }
    })
}

/// Sums of `[x - shift, (x - shift)**2]` (or `|x - shift|**2`) of the rows of every label, shape (k, z).
#[pyfunction]
fn egmm_label_stats<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    shift: PyReadonlyArray1<'py, f64>,
    labels: PyReadonlyArray1<'py, i64>,
    n_components: usize,
    covariance_type: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let dims = Dims::new(d, n_components, covariance(covariance_type)?);
    check_shape("shift", shift.shape(), &[d])?;
    check_shape("labels", labels.shape(), &[n])?;
    let (xs, sh, ls) = (
        slice(&x, "X")?,
        slice(&shift, "shift")?,
        slice(&labels, "labels")?,
    );
    if ls.iter().any(|&l| l < 0 || l as usize >= n_components) {
        return Err(PyValueError::new_err("labels must be in [0, n_components)"));
    }
    let sums = py.detach(|| egmm::label_stats(xs, n, sh, ls, dims));
    PyArray1::from_vec(py, sums).reshape([n_components, d + dims.s])
}

/// Pointer held by a `PyCapsule`, whatever its name.
fn capsule_pointer(capsule: &Bound<'_, PyAny>) -> PyResult<*mut std::ffi::c_void> {
    unsafe {
        if pyo3::ffi::PyCapsule_CheckExact(capsule.as_ptr()) == 0 {
            return Err(PyValueError::new_err("not a capsule"));
        }
        let name = pyo3::ffi::PyCapsule_GetName(capsule.as_ptr());
        let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name);
        if pointer.is_null() {
            return Err(PyErr::fetch(capsule.py()));
        }
        Ok(pointer)
    }
}

/// Use the BLAS that scipy ships for matrix products, like scikit-learn's `KMeans` does.
fn load_scipy_blas(py: Python<'_>) -> PyResult<()> {
    let capi = py
        .import("scipy.linalg.cython_blas")?
        .getattr("__pyx_capi__")?;
    let sgemm = capsule_pointer(&capi.get_item("sgemm")?)?;
    let dgemm = capsule_pointer(&capi.get_item("dgemm")?)?;
    unsafe { linalg::set_blas(sgemm, dgemm) };
    Ok(())
}

/// Switch matrix products between scipy's BLAS (`True`, the default when it loads) and pure Rust (`False`).
#[pyfunction]
fn use_blas(on: bool) {
    linalg::use_blas(on)
}

/// Whether matrix products currently go through scipy's BLAS.
#[pyfunction]
fn blas_active() -> bool {
    linalg::blas_active()
}

#[pymodule]
fn sklego_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Without scipy's BLAS the kernels fall back to the pure Rust `matrixmultiply`, which is slower but correct.
    let _ = load_scipy_blas(m.py());
    m.add_function(wrap_pyfunction!(use_blas, m)?)?;
    m.add_function(wrap_pyfunction!(blas_active, m)?)?;
    m.add_function(wrap_pyfunction!(emeans_fit_float32, m)?)?;
    m.add_function(wrap_pyfunction!(emeans_fit_float64, m)?)?;
    m.add_function(wrap_pyfunction!(emeans_nearest_float32, m)?)?;
    m.add_function(wrap_pyfunction!(emeans_nearest_float64, m)?)?;
    m.add_function(wrap_pyfunction!(emeans_distances_float32, m)?)?;
    m.add_function(wrap_pyfunction!(emeans_distances_float64, m)?)?;
    m.add_function(wrap_pyfunction!(egmm_fit, m)?)?;
    m.add_function(wrap_pyfunction!(egmm_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(egmm_label_stats, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
