use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spatialrs_core::{
    aggregation, autocorr, composition, gmm,
    interactions as srs_interactions,
    markers, neighbors, nmf, preprocess,
    rings, ripley, transitions,
    local_cor,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn to_py_err(e: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Convert an N×2 numpy f64 array to `Vec<[f64; 2]>` (no copy needed for coords).
fn coords_from_numpy(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 2]>> {
    let a = arr.as_array(); // numpy::ndarray 0.16 ArrayView
    if a.ncols() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coords array must have at least 2 columns",
        ));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

/// Convert numpy f64 array to ndarray 0.17 Array2<f64> (copies once via Vec).
///
/// This is necessary because the numpy crate (0.22) bundles ndarray 0.16, while
/// spatialrs-core uses ndarray 0.17. The two crate versions are distinct types,
/// so we bridge them through plain Vec + shape.
fn numpy_to_array2_f64(arr: PyReadonlyArray2<'_, f64>) -> ndarray::Array2<f64> {
    let view = arr.as_array();
    let shape = view.dim();
    let data: Vec<f64> = view.iter().copied().collect();
    ndarray::Array2::from_shape_vec(shape, data).expect("shape/data mismatch")
}

/// Convert numpy f64 array to ndarray 0.17 Array2<f32> (copies once via Vec).
fn numpy_to_array2_f32(arr: PyReadonlyArray2<'_, f64>) -> ndarray::Array2<f32> {
    let view = arr.as_array();
    let shape = view.dim();
    let data: Vec<f32> = view.iter().map(|&v| v as f32).collect();
    ndarray::Array2::from_shape_vec(shape, data).expect("shape/data mismatch")
}

/// Accept either a dense numpy array (float32 or float64) or a scipy sparse
/// matrix and return an ndarray 0.17 Array2<f32>.
///
/// For sparse matrices `.toarray()` is called automatically; the result may be
/// float32 or float64.
fn any_to_array2_f32(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<f32>> {
    // Dense float64
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(numpy_to_array2_f32(arr));
    }
    // Dense float32
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
        let view = arr.as_array();
        let shape = view.dim();
        let data: Vec<f32> = view.iter().copied().collect();
        return Ok(ndarray::Array2::from_shape_vec(shape, data).expect("shape/data mismatch"));
    }
    // Sparse scipy matrix — call .toarray() and accept either float dtype
    let dense = obj.call_method0("toarray").map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expression must be a numpy array (float32/float64) or a scipy sparse matrix",
        )
    })?;
    if let Ok(arr) = dense.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(numpy_to_array2_f32(arr));
    }
    if let Ok(arr) = dense.extract::<PyReadonlyArray2<'_, f32>>() {
        let view = arr.as_array();
        let shape = view.dim();
        let data: Vec<f32> = view.iter().copied().collect();
        return Ok(ndarray::Array2::from_shape_vec(shape, data).expect("shape/data mismatch"));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "sparse matrix .toarray() did not return a float32 or float64 numpy array",
    ))
}

/// Same as `any_to_array2_f32` but returns f64.
fn any_to_array2_f64(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<f64>> {
    // Dense float64
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(numpy_to_array2_f64(arr));
    }
    // Dense float32
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
        let view = arr.as_array();
        let shape = view.dim();
        let data: Vec<f64> = view.iter().map(|&v| v as f64).collect();
        return Ok(ndarray::Array2::from_shape_vec(shape, data).expect("shape/data mismatch"));
    }
    // Sparse scipy matrix — call .toarray() and accept either float dtype
    let dense = obj.call_method0("toarray").map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expression must be a numpy array (float32/float64) or a scipy sparse matrix",
        )
    })?;
    if let Ok(arr) = dense.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(numpy_to_array2_f64(arr));
    }
    if let Ok(arr) = dense.extract::<PyReadonlyArray2<'_, f32>>() {
        let view = arr.as_array();
        let shape = view.dim();
        let data: Vec<f64> = view.iter().map(|&v| v as f64).collect();
        return Ok(ndarray::Array2::from_shape_vec(shape, data).expect("shape/data mismatch"));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "sparse matrix .toarray() did not return a float32 or float64 numpy array",
    ))
}

/// Convert ndarray 0.17 Array2<f32> to a numpy 2-D array (copies once via Vec).
fn array2_f32_to_numpy<'py>(
    py: Python<'py>,
    arr: ndarray::Array2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let shape = arr.dim();
    let data: Vec<f32> = arr.into_raw_vec_and_offset().0;
    let np_arr = numpy::ndarray::Array2::from_shape_vec(shape, data).unwrap();
    PyArray2::from_owned_array_bound(py, np_arr)
}

/// Convert ndarray 0.17 Array2<f64> to a numpy 2-D array (copies once via Vec).
fn array2_f64_to_numpy<'py>(
    py: Python<'py>,
    arr: ndarray::Array2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let shape = arr.dim();
    let data: Vec<f64> = arr.into_raw_vec_and_offset().0;
    let np_arr = numpy::ndarray::Array2::from_shape_vec(shape, data).unwrap();
    PyArray2::from_owned_array_bound(py, np_arr)
}

// ─── graph construction ───────────────────────────────────────────────────────

/// Build a radius graph.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2), dtype float64
///     Spatial coordinates (x, y) for each cell.
/// barcodes : list[str]
///     Cell identifiers, length N.
/// radius : float
///     Search radius (same units as coords).
/// group : str, optional
///     Label written into every output row — useful when looping over samples.
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, cell_j, distance, group
///     Edges in both directions.  Pass directly to ``pd.DataFrame()``.
#[pyfunction]
#[pyo3(signature = (coords, barcodes, radius, group = ""))]
fn radius_graph<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records = neighbors::radius_graph(&c, &barcodes, radius, group).map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("cell_j", r.cell_j)?;
        d.set_item("distance", r.distance)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

/// Build a k-nearest-neighbour graph.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2), dtype float64
/// barcodes : list[str]
/// k : int  —  number of nearest neighbours per cell (self excluded)
/// group : str, optional
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, cell_j, distance, group
#[pyfunction]
#[pyo3(signature = (coords, barcodes, k, group = ""))]
fn knn_graph<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    k: usize,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records = neighbors::knn_graph(&c, &barcodes, k, group).map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("cell_j", r.cell_j)?;
        d.set_item("distance", r.distance)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

/// Compute per-cell neighbour count (degree) for a radius graph.
///
/// Useful as a QC metric for choosing an appropriate radius.
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, n_neighbors, group
#[pyfunction]
#[pyo3(signature = (coords, barcodes, radius, group = ""))]
fn graph_stats<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records =
        neighbors::compute_graph_stats(&c, &barcodes, radius, group).map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("n_neighbors", r.n_neighbors)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── cell-type interactions ───────────────────────────────────────────────────

/// Count cell-type pair interactions within a radius.
///
/// Each undirected pair is counted once (canonical a ≤ b alphabetical order).
///
/// Returns
/// -------
/// list[dict]  —  keys: group, cell_type_a, cell_type_b, count
#[pyfunction]
#[pyo3(signature = (coords, barcodes, cell_types, radius, group = ""))]
fn count_interactions<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    cell_types: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records =
        srs_interactions::count_interactions(&c, &barcodes, &cell_types, radius, group)
            .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("group", r.group)?;
        d.set_item("cell_type_a", r.cell_type_a)?;
        d.set_item("cell_type_b", r.cell_type_b)?;
        d.set_item("count", r.count)?;
        list.append(d)?;
    }
    Ok(list)
}

/// Permutation-based interaction enrichment test.
///
/// Shuffles cell-type labels ``n_perms`` times to build a null distribution,
/// then returns z-scores and empirical p-values for each cell-type pair.
///
/// Parameters
/// ----------
/// n_perms : int, default 1000
/// seed : int, default 42
///
/// Returns
/// -------
/// list[dict]  —  keys: group, cell_type_a, cell_type_b, observed,
///                expected_mean, expected_std, z_score, p_value
#[pyfunction]
#[pyo3(signature = (coords, barcodes, cell_types, radius, n_perms = 1000, seed = 42, group = ""))]
fn interaction_stats<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    cell_types: Vec<String>,
    radius: f64,
    n_perms: usize,
    seed: u64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records = srs_interactions::permute_interactions(
        &c,
        &barcodes,
        &cell_types,
        radius,
        n_perms,
        seed,
        group,
    )
    .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("group", r.group)?;
        d.set_item("cell_type_a", r.cell_type_a)?;
        d.set_item("cell_type_b", r.cell_type_b)?;
        d.set_item("observed", r.observed)?;
        d.set_item("expected_mean", r.expected_mean)?;
        d.set_item("expected_std", r.expected_std)?;
        d.set_item("z_score", r.z_score)?;
        d.set_item("p_value", r.p_value)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── neighbourhood composition ────────────────────────────────────────────────

/// Compute per-cell neighbourhood composition within a radius.
///
/// For each cell, returns the fraction of each cell type among its neighbours.
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, cell_type, fraction, group
#[pyfunction]
#[pyo3(signature = (coords, barcodes, cell_types, radius, group = ""))]
fn neighborhood_composition<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    cell_types: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records =
        composition::compute_composition(&c, &barcodes, &cell_types, radius, group)
            .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("cell_type", r.cell_type)?;
        d.set_item("fraction", r.fraction)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── spatial autocorrelation ──────────────────────────────────────────────────

/// Compute global Moran's I for each feature column.
///
/// Spatial weights are binary (1 = within ``radius``, 0 = beyond).
/// Significance via analytical approximation under normality.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2), dtype float64
/// values : np.ndarray, shape (N, F), dtype float64
///     Feature matrix — one column per feature (gene expression, PCA dim, etc.).
/// feature_names : list[str], length F
/// radius : float
///
/// Returns
/// -------
/// list[dict]  —  keys: feature, moran_i, expected_i, variance_i, z_score, group
#[pyfunction]
#[pyo3(signature = (coords, values, feature_names, radius, group = ""))]
fn morans<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    values: Bound<'py, PyAny>,
    feature_names: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let vals = any_to_array2_f64(&values)?;
    let records =
        autocorr::compute_morans_i(&c, &vals, &feature_names, radius, group)
            .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("feature", r.feature)?;
        d.set_item("moran_i", r.moran_i)?;
        d.set_item("expected_i", r.expected_i)?;
        d.set_item("variance_i", r.variance_i)?;
        d.set_item("z_score", r.z_score)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

/// Compute Local Moran's I (LISA) for each cell × feature.
///
/// p-values are BH-corrected within each feature.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2), dtype float64
/// barcodes : list[str]
/// values : np.ndarray, shape (N, F), dtype float64
/// feature_names : list[str], length F
/// radius : float
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, feature, local_i, z_score, p_value,
///                q_value_bh, group
#[pyfunction]
#[pyo3(signature = (coords, barcodes, values, feature_names, radius, group = ""))]
fn lisa<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    values: Bound<'py, PyAny>,
    feature_names: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let vals = any_to_array2_f64(&values)?;
    let records =
        autocorr::compute_local_morans_i(&c, &barcodes, &vals, &feature_names, radius, group)
            .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("feature", r.feature)?;
        d.set_item("local_i", r.local_i)?;
        d.set_item("z_score", r.z_score)?;
        d.set_item("p_value", r.p_value)?;
        d.set_item("q_value_bh", r.q_value_bh)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── NMF ─────────────────────────────────────────────────────────────────────

/// Factorise an expression matrix with NMF (Lee & Seung multiplicative updates).
///
/// Parameters
/// ----------
/// expression : np.ndarray, shape (N, G), dtype float64
///     Dense count or normalised matrix.  Cast to float32 internally.
/// barcodes : list[str], length N
/// gene_names : list[str], length G
/// n_components : int, default 10
/// max_iter : int, default 200
/// tol : float, default 1e-4
/// seed : int, default 42
///
/// Returns
/// -------
/// dict with keys:
///   W  : np.ndarray float32, shape (N, K)  —  cell loadings
///   H  : np.ndarray float32, shape (K, G)  —  gene loadings
///   n_iter : int
///   final_error : float
///   component_variances : list[float]  —  fractional variance per component
///   barcodes : list[str]
///   gene_names : list[str]
///
/// Example
/// -------
/// >>> res = spatialrs.nmf_factorize(expr, barcodes, genes, n_components=15)
/// >>> W = res["W"]   # shape (N, 15)
/// >>> H = res["H"]   # shape (15, G)
#[pyfunction]
#[pyo3(signature = (expression, barcodes, gene_names, n_components = 10, max_iter = 200, tol = 1e-4, seed = 42))]
fn nmf_factorize<'py>(
    py: Python<'py>,
    expression: Bound<'py, PyAny>,
    barcodes: Vec<String>,
    gene_names: Vec<String>,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let config = nmf::NmfConfig {
        n_components,
        max_iter,
        tol: tol as f32,
        seed,
        ..nmf::NmfConfig::default()
    };

    // Try to extract CSR sparse components first (memory-efficient path).
    // A scipy CSR matrix exposes .data, .indices, .indptr, and .shape.
    let result = if let (Ok(data_obj), Ok(idx_obj), Ok(ptr_obj), Ok(shape_obj)) = (
        expression.getattr("data"),
        expression.getattr("indices"),
        expression.getattr("indptr"),
        expression.getattr("shape"),
    ) {
        // Ensure it is in CSR format (not CSC, COO, etc.)
        let fmt: String = expression
            .getattr("format")
            .and_then(|f| f.extract::<String>())
            .unwrap_or_default();
        if fmt == "csr" {
            let shape: (usize, usize) = shape_obj.extract()?;
            let n_obs = shape.0;
            let n_var = shape.1;

            // Validate metadata lengths before entering Rust core
            if barcodes.len() != n_obs {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "barcodes length ({}) does not match expression rows ({})",
                    barcodes.len(), n_obs
                )));
            }
            if gene_names.len() != n_var {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "gene_names length ({}) does not match expression columns ({})",
                    gene_names.len(), n_var
                )));
            }

            // data may be float32 or float64 — normalise to f32
            let data_f32: Vec<f32> = if let Ok(a) = data_obj.extract::<PyReadonlyArray1<'_, f32>>() {
                a.as_slice()?.to_vec()
            } else if let Ok(a) = data_obj.extract::<PyReadonlyArray1<'_, f64>>() {
                a.as_slice()?.iter().map(|&v| v as f32).collect()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "sparse matrix .data must be float32 or float64",
                ));
            };

            // indices / indptr may be int32 or int64
            let indices: Vec<usize> = if let Ok(a) = idx_obj.extract::<PyReadonlyArray1<'_, i32>>() {
                a.as_slice()?.iter().map(|&v| v as usize).collect()
            } else {
                idx_obj.extract::<PyReadonlyArray1<'_, i64>>()?
                    .as_slice()?.iter().map(|&v| v as usize).collect()
            };
            let indptr: Vec<usize> = if let Ok(a) = ptr_obj.extract::<PyReadonlyArray1<'_, i32>>() {
                a.as_slice()?.iter().map(|&v| v as usize).collect()
            } else {
                ptr_obj.extract::<PyReadonlyArray1<'_, i64>>()?
                    .as_slice()?.iter().map(|&v| v as usize).collect()
            };

            nmf::run_nmf_sparse(&data_f32, &indices, &indptr, n_obs, n_var, &config)
                .map_err(to_py_err)?
        } else {
            // Not CSR — convert to CSR first via scipy, then retry recursively
            let csr = expression
                .call_method0("tocsr")
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    "sparse matrix could not be converted to CSR format",
                ))?;
            let expr_f32 = any_to_array2_f32(&csr)?;
            if barcodes.len() != expr_f32.nrows() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "barcodes length ({}) does not match expression rows ({})",
                    barcodes.len(), expr_f32.nrows()
                )));
            }
            if gene_names.len() != expr_f32.ncols() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "gene_names length ({}) does not match expression columns ({})",
                    gene_names.len(), expr_f32.ncols()
                )));
            }
            nmf::run_nmf(&expr_f32, &config).map_err(to_py_err)?
        }
    } else {
        // Dense path
        let expr_f32 = any_to_array2_f32(&expression)?;
        if barcodes.len() != expr_f32.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "barcodes length ({}) does not match expression rows ({})",
                barcodes.len(), expr_f32.nrows()
            )));
        }
        if gene_names.len() != expr_f32.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "gene_names length ({}) does not match expression columns ({})",
                gene_names.len(), expr_f32.ncols()
            )));
        }
        nmf::run_nmf(&expr_f32, &config).map_err(to_py_err)?
    };

    let out = PyDict::new_bound(py);
    out.set_item("W", array2_f32_to_numpy(py, result.w))?;
    out.set_item("H", array2_f32_to_numpy(py, result.h))?;
    out.set_item("n_iter", result.n_iter)?;
    out.set_item("final_error", result.final_error)?;
    out.set_item("component_variances", result.component_variances)?;
    out.set_item("barcodes", barcodes)?;
    out.set_item("gene_names", gene_names)?;
    Ok(out)
}

// ─── GMM ─────────────────────────────────────────────────────────────────────

/// Cluster cells into spatial niches using a Gaussian Mixture Model.
///
/// Uses EM with k-means++ initialisation and diagonal covariance.
///
/// Parameters
/// ----------
/// features : np.ndarray, shape (N, D), dtype float64
///     Input feature matrix (e.g. NMF W matrix, PCA, aggregated embeddings).
/// barcodes : list[str], length N
/// n_components : int, default 10
///     Number of mixture components (niches).
/// max_iter : int, default 200
/// tol : float, default 1e-6
///     Convergence tolerance on log-likelihood change.
/// seed : int, default 42
///
/// Returns
/// -------
/// dict with keys:
///   labels : list[int]   —  hard niche assignment per cell (0-indexed)
///   barcodes : list[str]
///   log_likelihood : float
///   bic : float
///   aic : float
///   n_iter : int
#[pyfunction]
#[pyo3(signature = (features, barcodes, n_components = 10, max_iter = 200, tol = 1e-6, seed = 42))]
fn gmm_cluster<'py>(
    py: Python<'py>,
    features: Bound<'py, PyAny>,
    barcodes: Vec<String>,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let feat = any_to_array2_f64(&features)?;
    if barcodes.len() != feat.nrows() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "barcodes length ({}) does not match features rows ({})",
            barcodes.len(), feat.nrows()
        )));
    }
    let config = gmm::GmmConfig {
        n_components,
        max_iter,
        tol,
        seed,
        ..gmm::GmmConfig::default()
    };
    let result = gmm::run_gmm(&feat, &config).map_err(to_py_err)?;
    let out = PyDict::new_bound(py);
    out.set_item("labels", result.labels)?;
    out.set_item("barcodes", barcodes)?;
    out.set_item("log_likelihood", result.log_likelihood)?;
    out.set_item("bic", result.bic)?;
    out.set_item("aic", result.aic)?;
    out.set_item("n_iter", result.n_iter)?;
    Ok(out)
}

// ─── preprocessing ────────────────────────────────────────────────────────────

/// Normalize each cell to a target total count sum.
///
/// Parameters
/// ----------
/// expression : np.ndarray, shape (N, G), dtype float64
/// target_sum : float, default 10000
///
/// Returns
/// -------
/// np.ndarray float32, shape (N, G)
#[pyfunction]
#[pyo3(signature = (expression, target_sum = 1e4))]
fn normalize_total<'py>(
    py: Python<'py>,
    expression: Bound<'py, PyAny>,
    target_sum: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut x = any_to_array2_f32(&expression)?;
    preprocess::normalize_total(&mut x, target_sum as f32);
    Ok(array2_f32_to_numpy(py, x))
}

/// Apply log(1 + x) element-wise.
///
/// Parameters
/// ----------
/// expression : np.ndarray, shape (N, G), dtype float64
///
/// Returns
/// -------
/// np.ndarray float32, shape (N, G)
#[pyfunction]
fn log1p<'py>(
    py: Python<'py>,
    expression: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut x = any_to_array2_f32(&expression)?;
    preprocess::log1p_transform(&mut x);
    Ok(array2_f32_to_numpy(py, x))
}

/// Return a boolean mask selecting cells that pass QC thresholds.
///
/// Parameters
/// ----------
/// expression : np.ndarray, shape (N, G), dtype float64
/// min_genes : int, optional  —  minimum number of detected genes
/// max_genes : int, optional  —  maximum number of detected genes
/// min_counts : float, optional  —  minimum total counts
/// max_counts : float, optional  —  maximum total counts
///
/// Returns
/// -------
/// list[bool], length N  —  True = passes filter
#[pyfunction]
#[pyo3(signature = (expression, min_genes = None, max_genes = None, min_counts = None, max_counts = None))]
fn filter_cells(
    expression: Bound<'_, PyAny>,
    min_genes: Option<usize>,
    max_genes: Option<usize>,
    min_counts: Option<f64>,
    max_counts: Option<f64>,
) -> PyResult<Vec<bool>> {
    let x = any_to_array2_f32(&expression)?;
    Ok(preprocess::filter_cells_mask(
        &x,
        min_genes,
        max_genes,
        min_counts.map(|v| v as f32),
        max_counts.map(|v| v as f32),
    ))
}

/// Return a boolean mask selecting genes expressed in at least `min_cells` cells.
///
/// Parameters
/// ----------
/// expression : np.ndarray, shape (N, G), dtype float64
/// min_cells : int, default 10
///
/// Returns
/// -------
/// list[bool], length G  —  True = passes filter
#[pyfunction]
#[pyo3(signature = (expression, min_cells = 10))]
fn filter_genes(
    expression: Bound<'_, PyAny>,
    min_cells: usize,
) -> PyResult<Vec<bool>> {
    let x = any_to_array2_f32(&expression)?;
    Ok(preprocess::filter_genes_mask(&x, min_cells))
}

/// Z-score each gene to mean=0, std=1.
///
/// Parameters
/// ----------
/// expression : np.ndarray, shape (N, G), dtype float64
/// max_value : float, optional  —  clip scaled values to [-max_value, max_value]
///
/// Returns
/// -------
/// np.ndarray float32, shape (N, G)
#[pyfunction]
#[pyo3(signature = (expression, max_value = None))]
fn scale<'py>(
    py: Python<'py>,
    expression: Bound<'py, PyAny>,
    max_value: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = any_to_array2_f32(&expression)?;
    let result = preprocess::scale(&x, max_value.map(|v| v as f32));
    Ok(array2_f32_to_numpy(py, result))
}

/// Identify highly variable genes (Seurat v1 / scanpy default method).
///
/// Bins genes by mean expression, then z-scores log(dispersion) within each bin.
///
/// Parameters
/// ----------
/// expression : np.ndarray, shape (N, G), dtype float64
/// gene_names : list[str], length G
/// n_top_genes : int, default 2000
/// n_bins : int, default 20
///
/// Returns
/// -------
/// list[dict]  —  keys: gene, mean, variance, dispersion, dispersion_norm,
///                highly_variable
///     Pass directly to ``pd.DataFrame()``.
#[pyfunction]
#[pyo3(signature = (expression, gene_names, n_top_genes = 2000, n_bins = 20))]
fn highly_variable_genes<'py>(
    py: Python<'py>,
    expression: Bound<'py, PyAny>,
    gene_names: Vec<String>,
    n_top_genes: usize,
    n_bins: usize,
) -> PyResult<Bound<'py, PyList>> {
    let x = any_to_array2_f32(&expression)?;
    let records = preprocess::highly_variable_genes(&x, &gene_names, n_top_genes, n_bins)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("gene", r.gene)?;
        d.set_item("mean", r.mean)?;
        d.set_item("variance", r.variance)?;
        d.set_item("dispersion", r.dispersion)?;
        d.set_item("dispersion_norm", r.dispersion_norm)?;
        d.set_item("highly_variable", r.highly_variable)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── helpers for new functions ────────────────────────────────────────────────

/// Accept a 1-D numpy array (f32 or f64) or a Python list of floats → Vec<f64>.
fn any_to_vec_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(a.as_array().iter().copied().collect());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, f32>>() {
        return Ok(a.as_array().iter().map(|&v| v as f64).collect());
    }
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a 1-D numpy array (float32/float64) or a list of floats",
    ))
}

/// Accept a 1-D numpy int array or Python list of ints → Vec<usize>.
fn any_to_vec_usize(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
        return a.as_array().iter().map(|&v| {
            if v < 0 { Err(pyo3::exceptions::PyValueError::new_err("negative niche label")) }
            else { Ok(v as usize) }
        }).collect();
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
        return a.as_array().iter().map(|&v| {
            if v < 0 { Err(pyo3::exceptions::PyValueError::new_err("negative niche label")) }
            else { Ok(v as usize) }
        }).collect();
    }
    if let Ok(v) = obj.extract::<Vec<usize>>() {
        return Ok(v);
    }
    // list of Python ints
    if let Ok(v) = obj.extract::<Vec<i64>>() {
        return v.iter().map(|&x| {
            if x < 0 { Err(pyo3::exceptions::PyValueError::new_err("negative niche label")) }
            else { Ok(x as usize) }
        }).collect();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "niche_labels must be a numpy int array or a list of non-negative integers",
    ))
}

// ─── spatial aggregation ──────────────────────────────────────────────────────

/// Distance-weighted aggregation of an embedding across spatial neighbours.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2), dtype float64
/// barcodes : list[str]
/// embedding : np.ndarray, shape (N, D), dtype float64
///     Embedding to aggregate (e.g. NMF W matrix, scVI, PCA).
/// graph_mode : str
///     ``"radius"`` or ``"knn"``.
/// graph_param : float
///     Radius in µm (for ``"radius"``) or number of neighbours (for ``"knn"``).
/// weighting : str, default ``"uniform"``
///     ``"uniform"``, ``"gaussian"``, ``"exponential"``, or
///     ``"inverse_distance"``.
/// weighting_param : float, optional
///     Sigma (gaussian), decay rate (exponential), or epsilon floor
///     (inverse_distance).
/// group : str, optional
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, dim, value, group
///     Long-format: one row per (cell, embedding dimension).
///     Pivot with ``pd.DataFrame(...).pivot(index="cell_i", columns="dim", values="value")``.
#[pyfunction]
#[pyo3(signature = (coords, barcodes, embedding, graph_mode, graph_param, weighting = "uniform", weighting_param = None, group = ""))]
fn aggregate<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    embedding: Bound<'py, PyAny>,
    graph_mode: &str,
    graph_param: f64,
    weighting: &str,
    weighting_param: Option<f64>,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let emb = any_to_array2_f64(&embedding)?;
    let graph = match graph_mode {
        "radius" => aggregation::GraphMode::Radius(graph_param),
        "knn" => {
            let k = graph_param as usize;
            if k == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "graph_param (k) must be >= 1 for knn mode",
                ));
            }
            aggregation::GraphMode::Knn(k)
        }
        other => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("graph_mode must be 'radius' or 'knn', got '{other}'"),
        )),
    };
    let wp = weighting_param.unwrap_or(1.0);
    let weight = match weighting {
        "uniform"          => aggregation::WeightingMode::Uniform,
        "gaussian"         => aggregation::WeightingMode::Gaussian { sigma: wp },
        "exponential"      => aggregation::WeightingMode::Exponential { decay: wp },
        "inverse_distance" => aggregation::WeightingMode::InverseDistance { epsilon: wp },
        other              => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("weighting must be 'uniform', 'gaussian', 'exponential', or 'inverse_distance', got '{other}'"),
        )),
    };
    let records = aggregation::aggregate_neighbors(&c, &barcodes, &emb, &graph, &weight, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("dim", r.dim)?;
        d.set_item("value", r.value)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── multiscale aggregation ───────────────────────────────────────────────────

/// Aggregate an embedding at multiple spatial scales and stack the results.
///
/// Implements the CellCharter-style niche-detection workflow: each cell's
/// feature vector becomes the concatenation of its own embedding (0-hop) plus
/// its neighbourhood-averaged embedding at each supplied radius.  Passing the
/// output directly to ``gmm_cluster`` yields tissue compartments rather than
/// cell-type clusters.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2), dtype float64
/// barcodes : list[str]
/// embedding : np.ndarray, shape (N, D), dtype float64
///     Per-cell embedding (e.g. NMF ``W`` matrix cast to float64).
/// radii : list[float]
///     Spatial radii at which to aggregate, in the same units as ``coords``.
///     E.g. ``[50.0, 150.0, 300.0]`` builds 3 neighbourhood scales.
/// include_self : bool, default True
///     Prepend the raw (0-hop) embedding as the first block.
/// weighting : str, default ``"gaussian"``
///     ``"uniform"``, ``"gaussian"``, ``"exponential"``, ``"inverse_distance"``.
/// weighting_param : float, optional
///     Sigma (gaussian), decay rate (exponential), or epsilon (inverse_distance).
///     Defaults to the smallest radius / 2 when unset.
/// group : str, optional
///
/// Returns
/// -------
/// np.ndarray float64, shape (N, D × n_blocks)
///     Ready to pass to ``gmm_cluster``.  n_blocks = len(radii) + (1 if
///     include_self else 0).
#[pyfunction]
#[pyo3(signature = (coords, barcodes, embedding, radii, include_self = true, weighting = "gaussian", weighting_param = None, group = ""))]
fn multiscale_aggregate<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    embedding: Bound<'py, PyAny>,
    radii: Vec<f64>,
    include_self: bool,
    weighting: &str,
    weighting_param: Option<f64>,
    group: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = group; // stored in output array metadata in future; unused for now
    let c = coords_from_numpy(coords)?;
    let emb = any_to_array2_f64(&embedding)?;
    let wp = weighting_param.unwrap_or_else(|| {
        radii.iter().cloned().fold(f64::INFINITY, f64::min) / 2.0
    });
    let weight = match weighting {
        "uniform"          => aggregation::WeightingMode::Uniform,
        "gaussian"         => aggregation::WeightingMode::Gaussian { sigma: wp },
        "exponential"      => aggregation::WeightingMode::Exponential { decay: wp },
        "inverse_distance" => aggregation::WeightingMode::InverseDistance { epsilon: wp },
        other              => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("weighting must be 'uniform', 'gaussian', 'exponential', or 'inverse_distance', got '{other}'"),
        )),
    };
    let result = aggregation::multiscale_aggregate(&c, &barcodes, &emb, &radii, include_self, &weight)
        .map_err(to_py_err)?;
    Ok(array2_f64_to_numpy(py, result))
}

// ─── niche markers ────────────────────────────────────────────────────────────

/// Find marker genes for each spatial niche (one-vs-rest Wilcoxon rank-sum).
///
/// Parameters
/// ----------
/// expression : array-like, shape (N, G)
///     Log-normalised dense expression matrix.
/// gene_names : list[str], length G
/// niche_labels : array-like of int, length N
///     Hard niche assignment per cell (0-indexed, as returned by
///     ``gmm_cluster``).
///
/// Returns
/// -------
/// list[dict]  —  keys: niche, gene, mean_niche, mean_rest, log2fc,
///                z_score, p_value, q_value_bh
#[pyfunction]
fn niche_markers<'py>(
    py: Python<'py>,
    expression: Bound<'py, PyAny>,
    gene_names: Vec<String>,
    niche_labels: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let x = any_to_array2_f32(&expression)?;
    let labels = any_to_vec_usize(&niche_labels)?;
    let n_niches = labels.iter().max().copied().unwrap_or(0) + 1;
    let records = markers::find_niche_markers(&x, &gene_names, &labels, n_niches)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("niche", r.niche)?;
        d.set_item("gene", r.gene)?;
        d.set_item("mean_niche", r.mean_niche)?;
        d.set_item("mean_rest", r.mean_rest)?;
        d.set_item("log2fc", r.log2fc)?;
        d.set_item("z_score", r.z_score)?;
        d.set_item("p_value", r.p_value)?;
        d.set_item("q_value_bh", r.q_value_bh)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── niche transitions ────────────────────────────────────────────────────────

/// Count niche co-occurrences within a radius (transition matrix).
///
/// Returns
/// -------
/// list[dict]  —  keys: niche_a, niche_b, count, fraction, group
#[pyfunction]
#[pyo3(signature = (coords, niche_labels, radius, group = ""))]
fn niche_transitions<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    niche_labels: Bound<'py, PyAny>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let labels = any_to_vec_usize(&niche_labels)?;
    let n_niches = labels.iter().max().copied().unwrap_or(0) + 1;
    let records = transitions::compute_transitions(&c, &labels, radius, n_niches, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("niche_a", r.niche_a)?;
        d.set_item("niche_b", r.niche_b)?;
        d.set_item("count", r.count)?;
        d.set_item("fraction", r.fraction)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

/// Permutation-based niche co-occurrence enrichment test.
///
/// Returns
/// -------
/// list[dict]  —  keys: niche_a, niche_b, observed, expected_mean,
///                expected_std, z_score, p_value, group
#[pyfunction]
#[pyo3(signature = (coords, niche_labels, radius, n_perms = 1000, seed = 42, group = ""))]
fn niche_transition_stats<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    niche_labels: Bound<'py, PyAny>,
    radius: f64,
    n_perms: usize,
    seed: u64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let labels = any_to_vec_usize(&niche_labels)?;
    let n_niches = labels.iter().max().copied().unwrap_or(0) + 1;
    let records = transitions::permute_transitions(&c, &labels, radius, n_niches, n_perms, seed, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("niche_a", r.niche_a)?;
        d.set_item("niche_b", r.niche_b)?;
        d.set_item("observed", r.observed)?;
        d.set_item("expected_mean", r.expected_mean)?;
        d.set_item("expected_std", r.expected_std)?;
        d.set_item("z_score", r.z_score)?;
        d.set_item("p_value", r.p_value)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── concentric rings ─────────────────────────────────────────────────────────

/// Compute cell-type composition in concentric distance rings around each cell.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2)
/// barcodes : list[str]
/// cell_types : list[str]
/// ring_edges : list[float]
///     Ring boundaries in µm, e.g. ``[0, 20, 50, 100, 200]``.
///     Produces ``len(ring_edges) - 1`` rings.
/// include_zeros : bool, default False
///     If True, emit rows for cell types with zero count in a ring.
/// group : str, optional
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, ring_inner, ring_outer, cell_type, count,
///                fraction, group
#[pyfunction]
#[pyo3(signature = (coords, barcodes, cell_types, ring_edges, include_zeros = false, group = ""))]
fn neighborhood_rings<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    cell_types: Vec<String>,
    ring_edges: Vec<f64>,
    include_zeros: bool,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records = rings::compute_rings(&c, &barcodes, &cell_types, &ring_edges, include_zeros, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("ring_inner", r.ring_inner)?;
        d.set_item("ring_outer", r.ring_outer)?;
        d.set_item("cell_type", r.cell_type)?;
        d.set_item("count", r.count)?;
        d.set_item("fraction", r.fraction)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── local correlation ────────────────────────────────────────────────────────

/// Compute per-cell Pearson correlation between two features within a radius.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2)
/// barcodes : list[str]
/// values_a, values_b : array-like, length N
///     Feature vectors (gene expression, NMF component, etc.).
/// feature_a, feature_b : str
///     Labels written into output rows.
/// radius : float
/// group : str, optional
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_i, feature_a, feature_b, local_r, n_neighbors,
///                group
#[pyfunction]
#[pyo3(signature = (coords, barcodes, values_a, values_b, feature_a, feature_b, radius, group = ""))]
fn local_correlation<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    barcodes: Vec<String>,
    values_a: Bound<'py, PyAny>,
    values_b: Bound<'py, PyAny>,
    feature_a: &str,
    feature_b: &str,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let va = any_to_vec_f64(&values_a)?;
    let vb = any_to_vec_f64(&values_b)?;
    let records = local_cor::compute_local_cor(&c, &barcodes, &va, &vb, feature_a, feature_b, radius, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_i", r.cell_i)?;
        d.set_item("feature_a", r.feature_a)?;
        d.set_item("feature_b", r.feature_b)?;
        d.set_item("local_r", r.local_r)?;
        d.set_item("n_neighbors", r.n_neighbors)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── Geary's C ────────────────────────────────────────────────────────────────

/// Compute Geary's C for each feature column.
///
/// C < 1 → positive spatial autocorrelation (similar values cluster).
/// C > 1 → negative spatial autocorrelation.
/// Equivalent to ``morans`` but more sensitive to local dissimilarity.
///
/// Returns
/// -------
/// list[dict]  —  keys: feature, geary_c, expected_c, variance_c, z_score,
///                group
#[pyfunction]
#[pyo3(signature = (coords, values, feature_names, radius, group = ""))]
fn geary<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    values: Bound<'py, PyAny>,
    feature_names: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let vals = any_to_array2_f64(&values)?;
    let records = autocorr::compute_gearys_c(&c, &vals, &feature_names, radius, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("feature", r.feature)?;
        d.set_item("geary_c", r.geary_c)?;
        d.set_item("expected_c", r.expected_c)?;
        d.set_item("variance_c", r.variance_c)?;
        d.set_item("z_score", r.z_score)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── bivariate Moran's I ──────────────────────────────────────────────────────

/// Compute bivariate Moran's I for all pairs of feature columns.
///
/// Measures spatial cross-correlation: are high values of feature A spatially
/// co-located with high values of feature B?
///
/// Returns
/// -------
/// list[dict]  —  keys: feature_a, feature_b, bivariate_i, z_score, group
#[pyfunction]
#[pyo3(signature = (coords, values, feature_names, radius, group = ""))]
fn bivariate_morans<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    values: Bound<'py, PyAny>,
    feature_names: Vec<String>,
    radius: f64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let vals = any_to_array2_f64(&values)?;
    let records = autocorr::compute_bivariate_morans_i(&c, &vals, &feature_names, radius, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("feature_a", r.feature_a)?;
        d.set_item("feature_b", r.feature_b)?;
        d.set_item("bivariate_i", r.bivariate_i)?;
        d.set_item("z_score", r.z_score)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── Ripley's K/L ─────────────────────────────────────────────────────────────

/// Compute Ripley's L(r) with Monte Carlo CSR confidence envelope.
///
/// Parameters
/// ----------
/// coords : np.ndarray, shape (N, 2)
/// cell_types : list[str]
/// target_type : str
///     Cell type to analyse.
/// radii : list[float]
///     Radii at which to evaluate K/L.
/// n_sims : int, default 199
///     Number of CSR simulations for the envelope.
/// seed : int, default 42
/// group : str, optional
///
/// Returns
/// -------
/// list[dict]  —  keys: cell_type, r, l_r, l_lo, l_hi, group
#[pyfunction]
#[pyo3(signature = (coords, cell_types, target_type, radii, n_sims = 199, seed = 42, group = ""))]
fn ripley_l<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    cell_types: Vec<String>,
    target_type: &str,
    radii: Vec<f64>,
    n_sims: usize,
    seed: u64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records = ripley::ripley_envelope(&c, &cell_types, target_type, &radii, n_sims, seed, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("cell_type", r.cell_type)?;
        d.set_item("r", r.r)?;
        d.set_item("l_r", r.l_r)?;
        d.set_item("l_lo", r.l_lo)?;
        d.set_item("l_hi", r.l_hi)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

/// Compute cross-Ripley L(r) with Monte Carlo CSR confidence envelope.
///
/// Measures whether cells of ``type_b`` are spatially attracted to or repelled
/// from cells of ``type_a``.
///
/// Returns
/// -------
/// list[dict]  —  keys: type_a, type_b, r, l_cross, l_lo, l_hi, group
#[pyfunction]
#[pyo3(signature = (coords, cell_types, type_a, type_b, radii, n_sims = 199, seed = 42, group = ""))]
fn cross_ripley_l<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    cell_types: Vec<String>,
    type_a: &str,
    type_b: &str,
    radii: Vec<f64>,
    n_sims: usize,
    seed: u64,
    group: &str,
) -> PyResult<Bound<'py, PyList>> {
    let c = coords_from_numpy(coords)?;
    let records = ripley::cross_ripley_envelope(&c, &cell_types, type_a, type_b, &radii, n_sims, seed, group)
        .map_err(to_py_err)?;
    let list = PyList::empty_bound(py);
    for r in records {
        let d = PyDict::new_bound(py);
        d.set_item("type_a", r.type_a)?;
        d.set_item("type_b", r.type_b)?;
        d.set_item("r", r.r)?;
        d.set_item("l_cross", r.l_cross)?;
        d.set_item("l_lo", r.l_lo)?;
        d.set_item("l_hi", r.l_hi)?;
        d.set_item("group", r.group)?;
        list.append(d)?;
    }
    Ok(list)
}

// ─── module ───────────────────────────────────────────────────────────────────

#[pymodule]
fn spatialrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // graph construction
    m.add_function(wrap_pyfunction!(radius_graph, m)?)?;
    m.add_function(wrap_pyfunction!(knn_graph, m)?)?;
    m.add_function(wrap_pyfunction!(graph_stats, m)?)?;
    // interactions
    m.add_function(wrap_pyfunction!(count_interactions, m)?)?;
    m.add_function(wrap_pyfunction!(interaction_stats, m)?)?;
    // composition
    m.add_function(wrap_pyfunction!(neighborhood_composition, m)?)?;
    // autocorrelation
    m.add_function(wrap_pyfunction!(morans, m)?)?;
    m.add_function(wrap_pyfunction!(lisa, m)?)?;
    m.add_function(wrap_pyfunction!(geary, m)?)?;
    m.add_function(wrap_pyfunction!(bivariate_morans, m)?)?;
    // dimensionality reduction / clustering
    m.add_function(wrap_pyfunction!(nmf_factorize, m)?)?;
    m.add_function(wrap_pyfunction!(gmm_cluster, m)?)?;
    // spatial aggregation
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(multiscale_aggregate, m)?)?;
    // niche analysis
    m.add_function(wrap_pyfunction!(niche_markers, m)?)?;
    m.add_function(wrap_pyfunction!(niche_transitions, m)?)?;
    m.add_function(wrap_pyfunction!(niche_transition_stats, m)?)?;
    // spatial patterns
    m.add_function(wrap_pyfunction!(neighborhood_rings, m)?)?;
    m.add_function(wrap_pyfunction!(local_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(ripley_l, m)?)?;
    m.add_function(wrap_pyfunction!(cross_ripley_l, m)?)?;
    // preprocessing
    m.add_function(wrap_pyfunction!(normalize_total, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(filter_cells, m)?)?;
    m.add_function(wrap_pyfunction!(filter_genes, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(highly_variable_genes, m)?)?;
    Ok(())
}
