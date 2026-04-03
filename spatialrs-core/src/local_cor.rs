use anyhow::{bail, Result};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct LocalCorRecord {
    pub cell_i: String,
    pub feature_a: String,
    pub feature_b: String,
    pub local_r: f64,
    pub n_neighbors: usize,
    pub group: String,
}

// ─── rstar integration ────────────────────────────────────────────────────────

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

// ─── Pearson r helper ─────────────────────────────────────────────────────────

fn pearson_r(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let cov: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - mean_a) * (bi - mean_b)).sum();
    let var_a: f64 = a.iter().map(|&ai| (ai - mean_a).powi(2)).sum();
    let var_b: f64 = b.iter().map(|&bi| (bi - mean_b).powi(2)).sum();
    if var_a < 1e-14 || var_b < 1e-14 {
        return f64::NAN;
    }
    (cov / (var_a * var_b).sqrt()).clamp(-1.0, 1.0)
}

// ─── main entry point ─────────────────────────────────────────────────────────

/// Compute per-cell local Pearson correlation between two features within a
/// radius neighbourhood.
///
/// Each cell's neighbourhood includes the focal cell itself (standard for local
/// statistics).  `local_r` is `f64::NAN` when `n_neighbors < 3` (including
/// focal cell) or when either feature has zero local variance.
pub fn compute_local_cor(
    coords: &[[f64; 2]],
    barcodes: &[String],
    values_a: &[f64],
    values_b: &[f64],
    feature_a: &str,
    feature_b: &str,
    radius: f64,
    group: &str,
) -> Result<Vec<LocalCorRecord>> {
    let n = coords.len();
    if n < 3 {
        bail!("need at least 3 cells for local correlation, got {n}");
    }
    if !radius.is_finite() || radius <= 0.0 {
        bail!("radius must be a finite value > 0");
    }
    if barcodes.len() != n {
        bail!("barcodes length ({}) != coords length ({n})", barcodes.len());
    }
    if values_a.len() != n {
        bail!("values_a length ({}) != coords length ({n})", values_a.len());
    }
    if values_b.len() != n {
        bail!("values_b length ({}) != coords length ({n})", values_b.len());
    }

    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);
    let r2 = radius * radius;

    let records: Vec<LocalCorRecord> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Neighbourhood includes focal cell (standard for local statistics)
            let nbrs: Vec<usize> = tree
                .locate_within_distance(coords[i], r2)
                .map(|p| p.index)
                .collect();
            let n_neighbors = nbrs.len();

            let local_r = if n_neighbors < 3 {
                f64::NAN
            } else {
                let a: Vec<f64> = nbrs.iter().map(|&j| values_a[j]).collect();
                let b: Vec<f64> = nbrs.iter().map(|&j| values_b[j]).collect();
                pearson_r(&a, &b)
            };

            LocalCorRecord {
                cell_i: barcodes[i].clone(),
                feature_a: feature_a.to_string(),
                feature_b: feature_b.to_string(),
                local_r,
                n_neighbors,
                group: group.to_string(),
            }
        })
        .collect();

    Ok(records)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{compute_local_cor, pearson_r};

    #[test]
    fn pearson_r_perfectly_correlated() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0]; // b = 2a → r = 1
        let r = pearson_r(&a, &b);
        assert!((r - 1.0).abs() < 1e-10, "expected r=1, got {r}");
    }

    #[test]
    fn pearson_r_anticorrelated() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let r = pearson_r(&a, &b);
        assert!((r + 1.0).abs() < 1e-10, "expected r=-1, got {r}");
    }

    #[test]
    fn pearson_r_zero_variance_returns_nan() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(pearson_r(&a, &b).is_nan());
    }

    #[test]
    fn local_cor_clustered_coexpression_positive() {
        // Two clusters: in cluster 1 both A and B are high; in cluster 2 both are low.
        // Cells within radius of their own cluster should have positive local r.
        let coords = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], // cluster 1
            [20.0, 0.0], [21.0, 0.0], [20.0, 1.0], [21.0, 1.0], // cluster 2
        ];
        let barcodes: Vec<String> = (0..8).map(|i| format!("c{i}")).collect();
        let values_a = vec![1.0, 2.0, 1.5, 2.5, 5.0, 6.0, 5.5, 6.5];
        let values_b = vec![1.0, 2.5, 1.2, 2.8, 5.5, 6.2, 5.8, 6.0];

        let records =
            compute_local_cor(&coords, &barcodes, &values_a, &values_b, "A", "B", 3.0, "test")
                .unwrap();
        assert_eq!(records.len(), 8);

        // All cells within a cluster should have non-NaN correlation
        for r in &records {
            assert!(!r.local_r.is_nan(), "expected non-NaN for cell {}", r.cell_i);
            assert!(r.local_r > 0.0, "expected positive correlation, got {}", r.local_r);
        }
    }

    #[test]
    fn local_cor_returns_nan_for_isolated_cell() {
        // Only 2 cells within each other's radius → n_neighbors < 3 → NaN
        let coords = [[0.0, 0.0], [0.5, 0.0], [100.0, 0.0]];
        let barcodes: Vec<String> = (0..3).map(|i| format!("c{i}")).collect();
        let values_a = vec![1.0, 2.0, 3.0];
        let values_b = vec![1.0, 2.0, 3.0];
        let records =
            compute_local_cor(&coords, &barcodes, &values_a, &values_b, "A", "B", 1.0, "test")
                .unwrap();
        // c2 has no neighbours within r=1 (only itself) → n_neighbors=1 < 3 → NaN
        let r2 = records.iter().find(|r| r.cell_i == "c2").unwrap();
        assert!(r2.local_r.is_nan(), "expected NaN for isolated cell");
    }
}
