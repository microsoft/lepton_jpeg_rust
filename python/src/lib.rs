use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Cursor;

#[pyfunction]
pub fn compress_bytes(py: Python, data: &[u8]) -> PyResult<Py<PyAny>> {
    let mut compressed = Vec::new();

    lepton_jpeg::encode_lepton(
        &mut Cursor::new(data),
        &mut Cursor::new(&mut compressed),
        &lepton_jpeg::EnabledFeatures::compat_lepton_vector_write(),
        &lepton_jpeg::DEFAULT_THREAD_POOL,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Compression failed: {}", e)))?;

    Ok(PyBytes::new(py, &compressed).into())
}

#[pyfunction]
pub fn decompress_bytes(py: Python, data: &[u8]) -> PyResult<Py<PyAny>> {
    let mut decompressed = Vec::new();

    lepton_jpeg::decode_lepton(
        &mut Cursor::new(data),
        &mut Cursor::new(&mut decompressed),
        &lepton_jpeg::EnabledFeatures::compat_lepton_vector_write(),
        &lepton_jpeg::DEFAULT_THREAD_POOL,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Decompression failed: {}", e))
    })?;

    Ok(PyBytes::new(py, &decompressed).into())
}

#[pymodule]
fn lepton_jpeg_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_bytes, m)?)?;
    Ok(())
}
