use eigenvalues::{Davidson, DavidsonCorrection, SpectrumTarget, lanczos::HermitianLanczos};
use nalgebra::DMatrix;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn eigdecomp<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray2<f64>)> {
    let matrix: DMatrix<f64> = matrix.readonly().as_matrix().into();

    let n = matrix.ncols();
    let eig = HermitianLanczos::new(
        matrix,
        n,
        SpectrumTarget::Lowest,
    )
    .unwrap();

    let eigvals = eig.eigenvalues.as_slice().to_vec().to_pyarray(py);
    let eigvects = eig.eigenvectors.to_pyarray(py);

    Ok((eigvals, eigvects))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn libeig_crate(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eigdecomp, m)?)?;
    Ok(())
}
