use anyhow::{bail, Result};
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;

// ─── normalize_total ──────────────────────────────────────────────────────────

/// Normalize each cell (row) so its total counts equal `target_sum`.
///
/// Cells with zero total counts are left unchanged.
/// Equivalent to `scanpy.pp.normalize_total`.
pub fn normalize_total(x: &mut Array2<f32>, target_sum: f32) {
    x.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let s: f32 = row.iter().sum();
            if s > 0.0 {
                let factor = target_sum / s;
                row.mapv_inplace(|v| v * factor);
            }
        });
}

// ─── log1p ────────────────────────────────────────────────────────────────────

/// Apply log(1 + x) to every element in-place.
/// Equivalent to `scanpy.pp.log1p`.
pub fn log1p_transform(x: &mut Array2<f32>) {
    x.par_mapv_inplace(|v| (v + 1.0).ln());
}

// ─── filter helpers ───────────────────────────────────────────────────────────

/// Return a per-cell boolean mask.
///
/// A cell passes if it satisfies ALL provided thresholds:
/// - `min_genes` / `max_genes`: number of genes with count > 0
/// - `min_counts` / `max_counts`: total UMI count
///
/// Equivalent to `scanpy.pp.filter_cells`.
pub fn filter_cells_mask(
    x: &Array2<f32>,
    min_genes: Option<usize>,
    max_genes: Option<usize>,
    min_counts: Option<f32>,
    max_counts: Option<f32>,
) -> Vec<bool> {
    x.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            let n_genes = row.iter().filter(|&&v| v > 0.0).count();
            let total: f32 = row.iter().sum();
            min_genes.map_or(true, |m| n_genes >= m)
                && max_genes.map_or(true, |m| n_genes <= m)
                && min_counts.map_or(true, |m| total >= m)
                && max_counts.map_or(true, |m| total <= m)
        })
        .collect()
}

/// Return a per-gene boolean mask.
///
/// A gene passes if it is non-zero in at least `min_cells` cells.
/// Equivalent to `scanpy.pp.filter_genes`.
pub fn filter_genes_mask(x: &Array2<f32>, min_cells: usize) -> Vec<bool> {
    (0..x.ncols())
        .into_par_iter()
        .map(|j| x.column(j).iter().filter(|&&v| v > 0.0).count() >= min_cells)
        .collect()
}

// ─── scale ────────────────────────────────────────────────────────────────────

/// Z-score each gene (column) to mean=0, std=1.
///
/// `max_value`: if set, clip values into `[-max_value, max_value]` after scaling.
/// Equivalent to `scanpy.pp.scale`.
pub fn scale(x: &Array2<f32>, max_value: Option<f32>) -> Array2<f32> {
    let (nrows, ncols) = x.dim();

    let params: Vec<(f32, f32)> = (0..ncols)
        .into_par_iter()
        .map(|j| {
            let col = x.column(j);
            let n = col.len() as f32;
            let mean = col.iter().sum::<f32>() / n;
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
            let std = var.sqrt().max(1e-8);
            (mean, std)
        })
        .collect();

    let mut result = Array2::<f32>::zeros((nrows, ncols));
    for j in 0..ncols {
        let (mean, std) = params[j];
        for (d, &s) in result.column_mut(j).iter_mut().zip(x.column(j).iter()) {
            let v = (s - mean) / std;
            *d = match max_value {
                Some(cap) => v.clamp(-cap, cap),
                None => v,
            };
        }
    }
    result
}

// ─── highly variable genes ────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct HvgRecord {
    pub gene: String,
    pub mean: f32,
    pub variance: f32,
    pub dispersion: f32,
    pub dispersion_norm: f32,
    pub highly_variable: bool,
}

/// Identify highly variable genes (Seurat v1 / scanpy default method).
///
/// Algorithm:
///   1. Compute mean and variance per gene.
///   2. Compute dispersion = variance / mean (Fano factor; 0 for zero-mean genes).
///   3. Bin genes by mean expression into `n_bins` equal-width bins.
///   4. Within each bin, z-score log(dispersion + 1).
///   5. Select the top `n_top_genes` by normalized dispersion.
///
/// Equivalent to `scanpy.pp.highly_variable_genes(flavor="seurat")`.
pub fn highly_variable_genes(
    x: &Array2<f32>,
    gene_names: &[String],
    n_top_genes: usize,
    n_bins: usize,
) -> Result<Vec<HvgRecord>> {
    let n_genes = x.ncols();
    if gene_names.len() != n_genes {
        bail!(
            "gene_names length ({}) != matrix columns ({})",
            gene_names.len(),
            n_genes
        );
    }
    if n_bins == 0 {
        bail!("n_bins must be > 0");
    }

    let n_cells = x.nrows() as f32;

    // Per-gene mean and variance
    let stats: Vec<(f32, f32)> = (0..n_genes)
        .into_par_iter()
        .map(|j| {
            let col = x.column(j);
            let mean = col.iter().sum::<f32>() / n_cells;
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n_cells;
            (mean, var)
        })
        .collect();

    let dispersions: Vec<f32> = stats
        .iter()
        .map(|&(mean, var)| if mean > 0.0 { var / mean } else { 0.0 })
        .collect();

    let means: Vec<f32> = stats.iter().map(|&(m, _)| m).collect();
    let min_mean = means.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_mean = means.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bin_width = ((max_mean - min_mean) / n_bins as f32).max(1e-8);

    let bin_ids: Vec<usize> = means
        .iter()
        .map(|&m| ((m - min_mean) / bin_width).floor() as usize)
        .collect();

    let log_disps: Vec<f32> = dispersions.iter().map(|&d| (d + 1.0).ln()).collect();

    // Per-bin mean and std of log-dispersion
    let mut bin_values: HashMap<usize, Vec<f32>> = HashMap::new();
    for (i, &bin) in bin_ids.iter().enumerate() {
        bin_values.entry(bin).or_default().push(log_disps[i]);
    }
    let bin_stats: HashMap<usize, (f32, f32)> = bin_values
        .iter()
        .map(|(&bin, vals)| {
            let n = vals.len() as f32;
            let mean = vals.iter().sum::<f32>() / n;
            let std = (vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n)
                .sqrt()
                .max(1e-8);
            (bin, (mean, std))
        })
        .collect();

    let disp_norm: Vec<f32> = (0..n_genes)
        .map(|i| {
            let (bin_mean, bin_std) = bin_stats[&bin_ids[i]];
            (log_disps[i] - bin_mean) / bin_std
        })
        .collect();

    // Top n_top_genes by normalized dispersion
    let mut ranked: Vec<usize> = (0..n_genes).collect();
    ranked.sort_by(|&a, &b| {
        disp_norm[b]
            .partial_cmp(&disp_norm[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_set: std::collections::HashSet<usize> = ranked.iter().take(n_top_genes).copied().collect();

    Ok((0..n_genes)
        .map(|i| HvgRecord {
            gene: gene_names[i].clone(),
            mean: stats[i].0,
            variance: stats[i].1,
            dispersion: dispersions[i],
            dispersion_norm: disp_norm[i],
            highly_variable: top_set.contains(&i),
        })
        .collect())
}
