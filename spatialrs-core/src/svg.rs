//! Spatially variable gene (SVG) detection.
//!
//! For each gene the following statistics are computed:
//!
//! * **Moran's I** and its analytical z-score (same as `autocorr::compute_morans_i`).
//! * **p-value** from the two-tailed normal approximation.
//! * **q-value (BH FDR)** — Benjamini-Hochberg correction applied *across all
//!   genes simultaneously*, so the ranking is valid for the whole transcriptome.
//! * **Spatial variance fraction** — fraction of each gene's total variance that
//!   is spatially structured.  For each cell, its neighbourhood mean expression is
//!   computed (neighbours within `radius`, self included); the variance of those
//!   neighbourhood means divided by the gene's total variance gives a value in
//!   [0, 1] where 1 means expression is perfectly smooth in space and 0 means all
//!   variance is local noise.
//!
//! Output rows are sorted by q-value (ascending), then z-score (descending).

use anyhow::{bail, Result};
use ndarray::Array2;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;

// ─── output record ────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct SvgRecord {
    pub gene: String,
    pub mean_expr: f64,
    pub moran_i: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub q_value_bh: f64,
    pub spatial_variance_fraction: f64,
    pub rank: usize,
    pub group: String,
}

// ─── spatial index ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct IndexedPoint {
    coords: [f64; 2],
    index: usize,
}

impl RTreeObject for IndexedPoint {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.coords)
    }
}

impl PointDistance for IndexedPoint {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dx = self.coords[0] - point[0];
        let dy = self.coords[1] - point[1];
        dx * dx + dy * dy
    }
}

// ─── normal CDF ───────────────────────────────────────────────────────────────

/// Two-tailed p-value from z-score via Abramowitz & Stegun 26.2.17.
/// Maximum absolute error < 7.5e-8.
fn two_tailed_p(z: f64) -> f64 {
    let x = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let poly = t * (0.319_381_530
        + t * (-0.356_563_782
            + t * (1.781_477_937
                + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt();
    let tail = (pdf * poly).clamp(0.0, 1.0);
    (2.0 * tail).min(1.0)
}

// ─── BH FDR correction ────────────────────────────────────────────────────────

/// Benjamini-Hochberg q-values.  Input: p-values in original gene order.
/// Returns q-values in the same order.
fn bh_correction(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    if n == 0 {
        return vec![];
    }

    // Sort indices by p-value ascending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // BH adjusted p-values (working backwards from the largest)
    let n_f = n as f64;
    let mut q_sorted = vec![0.0f64; n];
    let mut running_min = 1.0f64;
    for (rank_rev, &orig_idx) in order.iter().enumerate().rev() {
        let rank = rank_rev + 1; // 1-based
        let q = (p_values[orig_idx] * n_f / rank as f64).min(1.0);
        running_min = running_min.min(q);
        q_sorted[rank_rev] = running_min;
    }

    // Map back to original gene order
    let mut q_values = vec![0.0f64; n];
    for (rank_idx, &orig_idx) in order.iter().enumerate() {
        q_values[orig_idx] = q_sorted[rank_idx];
    }
    q_values
}

// ─── main function ────────────────────────────────────────────────────────────

/// Detect spatially variable genes.
///
/// `expression` is a dense N × G matrix (cells × genes).
/// All genes in `gene_names` are tested; use `filter_genes` upstream to
/// reduce the number of features if needed.
pub fn compute_svg(
    coords: &[[f64; 2]],
    expression: &Array2<f32>,
    gene_names: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<SvgRecord>> {
    let n = coords.len();
    let g = gene_names.len();

    if n < 4 {
        bail!("need at least 4 cells to compute SVG, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be finite and > 0");
    }
    if expression.nrows() != n {
        bail!(
            "expression rows ({}) != coords length ({n})",
            expression.nrows()
        );
    }
    if expression.ncols() != g {
        bail!(
            "expression cols ({}) != gene_names length ({g})",
            expression.ncols()
        );
    }

    // ── Build spatial index & edge list ──────────────────────────────────────
    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;

    // Upper-triangle edge pairs (undirected)
    let edge_pairs: Vec<(usize, usize)> = coords
        .par_iter()
        .enumerate()
        .flat_map(|(i, c)| {
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index > i)
                .map(|p| (i, p.index))
                .collect::<Vec<_>>()
        })
        .collect();

    let n_edges = edge_pairs.len();
    let s0 = 2.0 * n_edges as f64;

    if s0 == 0.0 {
        bail!("no neighbour edges found within radius {radius}");
    }

    let s1 = 2.0 * s0;
    let mut degrees = vec![0usize; n];
    for &(i, j) in &edge_pairs {
        degrees[i] += 1;
        degrees[j] += 1;
    }
    let s2: f64 = 4.0 * degrees.iter().map(|&d| (d as f64).powi(2)).sum::<f64>();

    let n_f = n as f64;
    let e_i = -1.0 / (n_f - 1.0);
    let denom = (n_f * n_f - 1.0) * s0 * s0;
    let var_i_base = if denom.abs() < 1e-14 {
        0.0
    } else {
        (n_f * n_f * s1 - n_f * s2 + 3.0 * s0 * s0) / denom - e_i * e_i
    };

    // ── Precompute adjacency lists for spatial variance fraction ─────────────
    // Include self in the neighbourhood mean for numerical stability.
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for &(i, j) in &edge_pairs {
        adj[i].push(j);
        adj[j].push(i);
    }
    for (i, nbrs) in adj.iter_mut().enumerate() {
        nbrs.push(i); // include self
    }

    // ── Per-gene statistics (parallel over genes) ─────────────────────────────
    let (moran_is, z_scores, p_values, svfs, mean_exprs): (
        Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>,
    ) = (0..g)
        .into_par_iter()
        .map(|f_idx| {
            let col: Vec<f64> = expression.column(f_idx).iter().map(|&v| v as f64).collect();
            let mean_x = col.iter().sum::<f64>() / n_f;
            let devs: Vec<f64> = col.iter().map(|&x| x - mean_x).collect();

            // Moran's I
            let numerator: f64 = edge_pairs.iter().map(|&(i, j)| devs[i] * devs[j]).sum::<f64>() * 2.0;
            let denominator: f64 = devs.iter().map(|&d| d * d).sum::<f64>();

            let moran_i = if denominator < 1e-14 {
                0.0
            } else {
                (n_f / s0) * (numerator / denominator)
            };

            let variance_i = var_i_base.max(0.0);
            let z_score = if variance_i < 1e-14 {
                0.0
            } else {
                (moran_i - e_i) / variance_i.sqrt()
            };

            let p_value = two_tailed_p(z_score);

            // Spatial variance fraction
            let nbr_means: Vec<f64> = adj
                .iter()
                .map(|nbrs| {
                    let s: f64 = nbrs.iter().map(|&k| col[k]).sum();
                    s / nbrs.len() as f64
                })
                .collect();
            let nbr_mean_of_means = nbr_means.iter().sum::<f64>() / n_f;
            let spatial_var: f64 = nbr_means
                .iter()
                .map(|&v| (v - nbr_mean_of_means).powi(2))
                .sum::<f64>()
                / n_f;
            let total_var = denominator / n_f; // = variance of raw expression
            let svf = if total_var < 1e-14 {
                0.0
            } else {
                (spatial_var / total_var).min(1.0)
            };

            (moran_i, z_score, p_value, svf, mean_x)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .fold(
            (Vec::with_capacity(g), Vec::with_capacity(g), Vec::with_capacity(g), Vec::with_capacity(g), Vec::with_capacity(g)),
            |(mut mi, mut zs, mut pv, mut sv, mut me), (m, z, p, s, e)| {
                mi.push(m); zs.push(z); pv.push(p); sv.push(s); me.push(e);
                (mi, zs, pv, sv, me)
            },
        );

    // ── BH FDR across all genes ───────────────────────────────────────────────
    let q_values = bh_correction(&p_values);

    // ── Build & sort output ───────────────────────────────────────────────────
    let mut records: Vec<SvgRecord> = (0..g)
        .map(|f_idx| SvgRecord {
            gene: gene_names[f_idx].clone(),
            mean_expr: mean_exprs[f_idx],
            moran_i: moran_is[f_idx],
            z_score: z_scores[f_idx],
            p_value: p_values[f_idx],
            q_value_bh: q_values[f_idx],
            spatial_variance_fraction: svfs[f_idx],
            rank: 0,
            group: group.to_string(),
        })
        .collect();

    // Sort: q ascending, then z_score descending
    records.sort_by(|a, b| {
        a.q_value_bh
            .partial_cmp(&b.q_value_bh)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.z_score
                    .partial_cmp(&a.z_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    for (rank, r) in records.iter_mut().enumerate() {
        r.rank = rank + 1;
    }

    Ok(records)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn grid_coords(nx: usize, ny: usize, spacing: f64) -> Vec<[f64; 2]> {
        (0..nx)
            .flat_map(|x| (0..ny).map(move |y| [x as f64 * spacing, y as f64 * spacing]))
            .collect()
    }

    #[test]
    fn detects_spatially_structured_gene() {
        let coords = grid_coords(8, 8, 10.0); // 64 cells
        let n = coords.len();
        let mut expr = Array2::<f32>::zeros((n, 2));
        // gene 0: perfectly spatially structured (x-gradient)
        for (i, c) in coords.iter().enumerate() {
            expr[[i, 0]] = c[0] as f32;
        }
        // gene 1: random noise (not spatial)
        let mut rng_state = 12345u64;
        for i in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            expr[[i, 1]] = ((rng_state >> 33) as f32) / u32::MAX as f32;
        }
        let gene_names = vec!["spatial_gene".to_string(), "random_gene".to_string()];
        let result = compute_svg(&coords, &expr, &gene_names, 15.0, "test").unwrap();
        let spatial = result.iter().find(|r| r.gene == "spatial_gene").unwrap();
        let random = result.iter().find(|r| r.gene == "random_gene").unwrap();
        assert!(spatial.moran_i > random.moran_i, "spatial gene should have higher Moran's I");
        assert!(spatial.z_score > 0.0);
        assert!(spatial.spatial_variance_fraction > random.spatial_variance_fraction);
    }

    #[test]
    fn q_values_are_valid() {
        let coords = grid_coords(6, 6, 10.0);
        let n = coords.len();
        let expr = Array2::<f32>::ones((n, 5));
        let names: Vec<String> = (0..5).map(|i| format!("g{i}")).collect();
        let result = compute_svg(&coords, &expr, &names, 15.0, "").unwrap();
        for r in &result {
            assert!(r.q_value_bh >= 0.0 && r.q_value_bh <= 1.0);
            assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
        }
    }

    #[test]
    fn two_tailed_p_sanity() {
        // z=0 → p≈1, z=1.96 → p≈0.05, z=3.29 → p≈0.001
        assert!((two_tailed_p(0.0) - 1.0).abs() < 0.01);
        assert!((two_tailed_p(1.96) - 0.05).abs() < 0.005);
        assert!((two_tailed_p(3.29) - 0.001).abs() < 0.0005);
    }
}
