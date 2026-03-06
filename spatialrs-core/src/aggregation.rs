use anyhow::{bail, Result};
use ndarray::Array2;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;

// ─── types ────────────────────────────────────────────────────────────────────

pub enum WeightingMode {
    Uniform,
    Gaussian { sigma: f64 },
    Exponential { decay: f64 },
    InverseDistance { epsilon: f64 },
}

pub enum GraphMode {
    Radius(f64),
    Knn(usize),
}

#[derive(Serialize)]
pub struct AggregationRecord {
    pub cell_i: String,
    pub dim: usize,
    pub value: f64,
    pub group: String,
}

// ─── rstar integration ────────────────────────────────────────────────────────

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

// ─── main function ────────────────────────────────────────────────────────────

/// For each cell, aggregate its neighbours' embedding rows using distance-weighted
/// averaging.  Outputs long-format records: one row per (cell, embedding dim).
pub fn aggregate_neighbors(
    coords: &[[f64; 2]],
    barcodes: &[String],
    embedding: &Array2<f64>,
    graph: &GraphMode,
    weighting: &WeightingMode,
    group: &str,
) -> Result<Vec<AggregationRecord>> {
    validate_aggregation_inputs(coords, barcodes, embedding, graph)?;

    let n = coords.len();
    let dim = embedding.ncols();

    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint {
            coords: c,
            index: i,
        })
        .collect();

    let tree = RTree::bulk_load(points);

    let records: Vec<AggregationRecord> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let c = coords[i];
            let neighbors: Vec<(usize, f64)> = find_neighbors(&tree, &c, i, graph);

            let mut agg = vec![0.0f64; dim];

            if neighbors.is_empty() {
                // Isolated cell — emit zeros
            } else {
                let weights: Vec<f64> = neighbors
                    .iter()
                    .map(|&(_, d)| compute_weight(d, weighting))
                    .collect();
                let w_sum: f64 = weights.iter().sum();

                for (k, (&(j, _), &w)) in neighbors.iter().zip(weights.iter()).enumerate() {
                    let _ = k;
                    let row = embedding.row(j);
                    for d in 0..dim {
                        agg[d] += w * row[d] / w_sum;
                    }
                }
            }

            agg.into_iter()
                .enumerate()
                .map(|(d, v)| AggregationRecord {
                    cell_i: barcodes[i].clone(),
                    dim: d,
                    value: v,
                    group: group.to_string(),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(records)
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn find_neighbors(
    tree: &RTree<IndexedPoint>,
    c: &[f64; 2],
    self_i: usize,
    graph: &GraphMode,
) -> Vec<(usize, f64)> {
    match graph {
        GraphMode::Radius(r) => {
            let r2 = r * r;
            tree.locate_within_distance(*c, r2)
                .filter(|p| p.index != self_i)
                .map(|p| (p.index, p.distance_2(c).sqrt()))
                .collect()
        }
        GraphMode::Knn(k) => tree
            .nearest_neighbor_iter_with_distance_2(c)
            .filter(|(p, _)| p.index != self_i)
            .take(*k)
            .map(|(p, d2)| (p.index, d2.sqrt()))
            .collect(),
    }
}

fn compute_weight(d: f64, mode: &WeightingMode) -> f64 {
    match mode {
        WeightingMode::Uniform => 1.0,
        WeightingMode::Gaussian { sigma } => (-d * d / (2.0 * sigma * sigma)).exp(),
        WeightingMode::Exponential { decay } => (-decay * d).exp(),
        WeightingMode::InverseDistance { epsilon } => 1.0 / (d + epsilon),
    }
}

fn validate_aggregation_inputs(
    coords: &[[f64; 2]],
    barcodes: &[String],
    embedding: &Array2<f64>,
    graph: &GraphMode,
) -> Result<()> {
    if coords.len() != barcodes.len() {
        bail!(
            "coords length ({}) does not match barcodes length ({})",
            coords.len(),
            barcodes.len()
        );
    }
    if coords.len() != embedding.nrows() {
        bail!(
            "coords length ({}) does not match embedding rows ({})",
            coords.len(),
            embedding.nrows()
        );
    }
    if let GraphMode::Radius(radius) = graph {
        if !radius.is_finite() || *radius <= 0.0 {
            bail!("radius must be a finite value > 0");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{aggregate_neighbors, GraphMode, WeightingMode};
    use ndarray::arr2;

    #[test]
    fn aggregate_rejects_non_positive_radius() {
        let coords = [[0.0, 0.0], [1.0, 1.0]];
        let barcodes = vec!["a".to_string(), "b".to_string()];
        let embedding = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let err = match aggregate_neighbors(
            &coords,
            &barcodes,
            &embedding,
            &GraphMode::Radius(0.0),
            &WeightingMode::Uniform,
            "g",
        ) {
            Ok(_) => panic!("expected invalid radius error"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("radius must be a finite value > 0"));
    }
}
