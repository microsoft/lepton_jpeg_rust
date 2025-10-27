use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::io::Cursor;

#[pyfunction]
#[pyo3(signature = (data, config=None))]
pub fn compress_bytes(
    py: Python,
    data: &[u8],
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let mut compressed = Vec::new();

    let mut features = lepton_jpeg::EnabledFeatures::compat_lepton_vector_write();

    if let Some(cfg) = config {
        for (key, value) in cfg.iter() {
            let key_str: &str = key.extract()?;
            match key_str {
                "max_jpeg_width" => {
                    let val: u32 = value.extract()?;
                    features.max_jpeg_width = val;
                }
                "max_jpeg_height" => {
                    let val: u32 = value.extract()?;
                    features.max_jpeg_height = val;
                }
                "progressive" => {
                    let val: bool = value.extract()?;
                    features.progressive = val;
                }
                "reject_dqts_with_zeros" => {
                    let val: bool = value.extract()?;
                    features.reject_dqts_with_zeros = val;
                }
                "max_partitions" => {
                    let val: u32 = value.extract()?;
                    features.max_threads = val;
                }
                "max_jpeg_file_size" => {
                    let val: u32 = value.extract()?;
                    features.max_jpeg_file_size = val;
                }
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown configuration key: {}",
                        key_str
                    )));
                }
            }
        }
    }

    lepton_jpeg::encode_lepton(
        &mut Cursor::new(data),
        &mut Cursor::new(&mut compressed),
        &features,
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
