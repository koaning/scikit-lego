# sklego-rust

Compiled kernels behind `backend="rust"` in `sklego.cluster.EMeans` and `sklego.mixture.EGaussianMixture`.

The Python estimators keep validation, the scaling-law defaults and the initialisation. This crate runs the training
loop (shuffling, mini-batch updates, revivals, final pass) and the full-data passes (`predict`, `transform`, `score`,
`score_samples`, `predict_proba`). Matrix products go through the BLAS that scipy ships (`scipy.linalg.cython_blas`,
as scikit-learn's `KMeans` does), with the pure Rust `matrixmultiply` crate as a fallback. Everything else runs on a
rayon thread pool, sized by `RAYON_NUM_THREADS`.

Only dense float arrays are supported: float32 and float64 for `EMeans`, float64 for `EGaussianMixture`.

## Install

Needs a Rust toolchain (https://rustup.rs). From the root of the scikit-lego repository:

```bash
uv pip install ./rust          # or: pip install ./rust
```

For development, `maturin develop --release -m rust/Cargo.toml` rebuilds in place.

## Test

```bash
cargo test --manifest-path rust/Cargo.toml        # Rust unit tests
pytest -m rust                                     # Python tests of the rust backend, against the NumPy one
```

## Layout

- `src/lib.rs`: PyO3 bindings. These are internal to scikit-lego and expect C-contiguous arrays and resolved
  parameters.
- `src/emeans.rs`: the e-means loop and nearest-centroid passes, generic over f32/f64.
- `src/egmm.rs`: the e-GMM loop and density evaluation.
- `src/linalg.rs`: GEMM dispatch (scipy BLAS or `matrixmultiply`) and the RNG (xoshiro256++).

The random numbers differ from NumPy's, so a `random_state` gives a different (equally good) model on each backend.
Configurations that do not depend on randomness (fixed init, one batch per epoch, no revival) match the NumPy backend
to about 1e-10.
