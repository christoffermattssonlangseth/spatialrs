use anyhow::{bail, Context, Result};
use hdf5_metno as hdf5;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

/// In-memory representation of the data we need from an .h5ad file.
pub struct AnnData {
    /// Cell barcodes / obs index, length N.
    pub obs_names: Vec<String>,
    /// Spatial coordinates, shape N×2 (x, y).
    pub coordinates: Array2<f64>,
    /// Obs columns requested by the caller (column name → per-cell string values).
    pub obs: HashMap<String, Vec<String>>,
}

/// Read an .h5ad file and return an `AnnData` containing the obs index,
/// spatial coordinates, and the requested obs columns.
pub fn read_h5ad(path: &Path, obs_cols: &[&str]) -> Result<AnnData> {
    let file = hdf5::File::open(path)
        .with_context(|| format!("cannot open {:?}", path))?;

    let obs_names = read_obs_names(&file)?;
    let coordinates = read_spatial(&file)?;

    if obs_names.len() != coordinates.nrows() {
        bail!(
            "obs_names length ({}) != coordinate rows ({})",
            obs_names.len(),
            coordinates.nrows()
        );
    }

    let mut obs = HashMap::new();
    for col in obs_cols {
        let values = read_obs_column(&file, col)
            .with_context(|| format!("reading obs column '{col}'"))?;
        obs.insert(col.to_string(), values);
    }

    Ok(AnnData { obs_names, coordinates, obs })
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn read_obs_names(file: &hdf5::File) -> Result<Vec<String>> {
    let obs = file.group("obs").context("no 'obs' group")?;

    // Strategy 1: dataset at obs/_index
    if let Ok(ds) = obs.dataset("_index") {
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading obs/_index");
    }

    // Strategy 2: dataset at obs/index
    if let Ok(ds) = obs.dataset("index") {
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .context("reading obs/index");
    }

    // Strategy 3: check _index attribute on the obs group for the real dataset name
    if let Ok(attr) = obs.attr("_index") {
        let name: hdf5::types::VarLenUnicode = attr.read_scalar()
            .context("reading obs _index attribute")?;
        let ds = obs.dataset(name.as_str())
            .with_context(|| format!("obs dataset '{}' (from _index attr)", name))?;
        return ds.read_1d::<hdf5::types::VarLenUnicode>()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .with_context(|| format!("reading obs/{}", name));
    }

    bail!("could not locate obs index in the h5ad file")
}

fn read_spatial(file: &hdf5::File) -> Result<Array2<f64>> {
    let obsm = file.group("obsm").context("no 'obsm' group")?;

    for key in &["spatial", "X_spatial"] {
        if let Ok(ds) = obsm.dataset(key) {
            // Try f64 first, fall back to f32
            if let Ok(arr) = ds.read_2d::<f64>() {
                return slice_two_cols_f64(arr);
            }
            if let Ok(arr) = ds.read_2d::<f32>() {
                let arr_f64 = arr.mapv(|v| v as f64);
                return slice_two_cols_f64(arr_f64);
            }
            bail!("obsm/{key} is not f32 or f64");
        }
    }

    bail!("no spatial coordinates found under obsm/spatial or obsm/X_spatial")
}

fn slice_two_cols_f64(arr: Array2<f64>) -> Result<Array2<f64>> {
    if arr.ncols() < 2 {
        bail!("spatial array has fewer than 2 columns");
    }
    Ok(arr.slice(ndarray::s![.., 0..2]).to_owned())
}

fn read_obs_column(file: &hdf5::File, col: &str) -> Result<Vec<String>> {
    let obs = file.group("obs")?;

    // Try categorical encoding first: obs/{col}/codes + obs/{col}/categories
    if let Ok(grp) = obs.group(col) {
        if let (Ok(codes_ds), Ok(cats_ds)) = (grp.dataset("codes"), grp.dataset("categories")) {
            let categories: Vec<String> = cats_ds
                .read_1d::<hdf5::types::VarLenUnicode>()
                .context("reading categories")?
                .iter()
                .map(|s| s.to_string())
                .collect();

            // codes can be i8, i16, or i32 depending on the writer
            let n = codes_ds.size();
            let codes: Vec<i32> = if let Ok(c) = codes_ds.read_1d::<i8>() {
                c.iter().map(|&v| v as i32).collect()
            } else if let Ok(c) = codes_ds.read_1d::<i16>() {
                c.iter().map(|&v| v as i32).collect()
            } else {
                codes_ds.read_1d::<i32>()
                    .with_context(|| format!("reading codes for obs/{col}"))?
                    .iter()
                    .map(|&v| v)
                    .collect()
            };

            if codes.len() != n {
                bail!("code length mismatch for obs/{col}");
            }

            return codes
                .iter()
                .map(|&c| {
                    categories
                        .get(c as usize)
                        .cloned()
                        .with_context(|| format!("code {c} out of range for obs/{col}"))
                })
                .collect();
        }
    }

    // Fallback: direct VarLenUnicode dataset at obs/{col}
    let ds = obs.dataset(col)
        .with_context(|| format!("obs/{col} not found"))?;
    ds.read_1d::<hdf5::types::VarLenUnicode>()
        .map(|a| a.iter().map(|s| s.to_string()).collect())
        .with_context(|| format!("reading obs/{col} as VarLenUnicode"))
}
