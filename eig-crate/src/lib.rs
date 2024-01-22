use nalgebra::DMatrix;
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyArray1, PyReadonlyArray2};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn eigdecomp<'py>(py: Python<'py>, matrix: PyReadonlyArray2<'py, f64>) -> PyResult<(&'py PyArray1<f64>, &'py PyArray2<f64>)> {
    let matrix: DMatrix<f64> = matrix.readonly().try_as_matrix().unwrap();
    todo!()
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn libeig_crate(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eigdecomp, m)?)?;
    Ok(())
}
