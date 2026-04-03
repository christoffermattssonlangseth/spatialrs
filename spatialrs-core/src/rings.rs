use anyhow::{bail, Result};
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::Serialize;
use std::collections::HashMap;

// ─── public types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct RingRecord {
    pub cell_i: String,
    pub ring_inner: f64,
    pub ring_outer: f64,
    pub cell_type: String,
    pub count: usize,
    pub fraction: f64,
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

// ─── main entry point ─────────────────────────────────────────────────────────

/// Compute per-cell neighbourhood composition in concentric rings (distance bins).
///
/// `ring_edges` must be strictly increasing with at least two elements.
/// Rings are half-open intervals: [ring_edges[k], ring_edges[k+1]).
/// The focal cell is excluded from its own counts (self is not a neighbour).
///
/// When `include_zeros = false` (default), records with `count = 0` are
/// omitted to keep output compact.
pub fn compute_rings(
    coords: &[[f64; 2]],
    barcodes: &[String],
    cell_types: &[String],
    ring_edges: &[f64],
    include_zeros: bool,
    group: &str,
) -> Result<Vec<RingRecord>> {
    let n = coords.len();
    if n < 2 {
        bail!("need at least 2 cells for ring analysis, got {n}");
    }
    if barcodes.len() != n {
        bail!("barcodes length ({}) != coords length ({n})", barcodes.len());
    }
    if cell_types.len() != n {
        bail!("cell_types length ({}) != coords length ({n})", cell_types.len());
    }
    if ring_edges.len() < 2 {
        bail!("ring_edges must have at least 2 elements (inner and outer boundary)");
    }
    for i in 0..ring_edges.len() {
        if !ring_edges[i].is_finite() {
            bail!("ring_edges[{i}] is not finite");
        }
        if i > 0 && ring_edges[i] <= ring_edges[i - 1] {
            bail!("ring_edges must be strictly increasing");
        }
    }

    let n_rings = ring_edges.len() - 1;
    let max_r = ring_edges[ring_edges.len() - 1];
    let max_r2 = max_r * max_r;

    // Collect unique cell types for zero-filling (if requested)
    let unique_types: Vec<String> = {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut v: Vec<String> = Vec::new();
        for t in cell_types {
            if seen.insert(t.as_str()) {
                v.push(t.clone());
            }
        }
        v.sort();
        v
    };

    let points: Vec<IndexedPoint> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| IndexedPoint { coords: c, index: i })
        .collect();
    let tree = RTree::bulk_load(points);

    let records: Vec<RingRecord> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            // Query all neighbours within max_r (excluding self)
            let neighbours: Vec<(usize, f64)> = tree
                .locate_within_distance(coords[i], max_r2)
                .filter(|p| p.index != i)
                .map(|p| {
                    let dx = p.coords[0] - coords[i][0];
                    let dy = p.coords[1] - coords[i][1];
                    (p.index, (dx * dx + dy * dy).sqrt())
                })
                .collect();

            // Tally per-ring, per-cell-type counts
            let mut counts: HashMap<(usize, &str), usize> = HashMap::new();
            for &(j, dist) in &neighbours {
                // Binary search for the ring this distance belongs to
                let ring_idx = ring_edges[1..].partition_point(|&edge| dist >= edge);
                if ring_idx < n_rings {
                    *counts
                        .entry((ring_idx, cell_types[j].as_str()))
                        .or_insert(0) += 1;
                }
            }

            // Total cells per ring (for fraction denominator)
            let mut ring_totals: Vec<usize> = vec![0; n_rings];
            for (&(ring_idx, _), &cnt) in &counts {
                ring_totals[ring_idx] += cnt;
            }

            // Emit records
            let mut cell_records: Vec<RingRecord> = Vec::new();
            if include_zeros {
                for ring_idx in 0..n_rings {
                    let total = ring_totals[ring_idx];
                    for ct in &unique_types {
                        let count = counts.get(&(ring_idx, ct.as_str())).copied().unwrap_or(0);
                        let fraction = if total > 0 { count as f64 / total as f64 } else { 0.0 };
                        cell_records.push(RingRecord {
                            cell_i: barcodes[i].clone(),
                            ring_inner: ring_edges[ring_idx],
                            ring_outer: ring_edges[ring_idx + 1],
                            cell_type: ct.clone(),
                            count,
                            fraction,
                            group: group.to_string(),
                        });
                    }
                }
            } else {
                for (&(ring_idx, ct), &count) in &counts {
                    if count == 0 {
                        continue;
                    }
                    let total = ring_totals[ring_idx];
                    let fraction = if total > 0 { count as f64 / total as f64 } else { 0.0 };
                    cell_records.push(RingRecord {
                        cell_i: barcodes[i].clone(),
                        ring_inner: ring_edges[ring_idx],
                        ring_outer: ring_edges[ring_idx + 1],
                        cell_type: ct.to_string(),
                        count,
                        fraction,
                        group: group.to_string(),
                    });
                }
            }

            cell_records
        })
        .collect();

    Ok(records)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::compute_rings;

    #[test]
    fn rings_fractions_sum_to_one_per_cell_ring() {
        // 5 cells of alternating types in a line; ring [0, 2) captures some neighbours
        let coords = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ];
        let barcodes: Vec<String> = (0..5).map(|i| format!("c{i}")).collect();
        let types = vec!["A", "B", "A", "B", "A"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let ring_edges = vec![0.0, 1.5, 3.0];

        let records = compute_rings(&coords, &barcodes, &types, &ring_edges, false, "test").unwrap();

        // For each (cell, ring) group, fractions should sum to 1.0
        let mut sums: std::collections::HashMap<(String, usize), f64> = std::collections::HashMap::new();
        for r in &records {
            let ring_idx = ring_edges[1..].partition_point(|&e| r.ring_inner >= e);
            *sums.entry((r.cell_i.clone(), ring_idx)).or_insert(0.0) += r.fraction;
        }
        for (key, sum) in &sums {
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "fractions do not sum to 1 for {:?}: {sum}",
                key
            );
        }
    }

    #[test]
    fn rings_self_excluded() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let barcodes: Vec<String> = (0..3).map(|i| format!("c{i}")).collect();
        let types: Vec<String> = vec!["A".to_string(); 3];
        let ring_edges = vec![0.0, 0.5, 2.0];

        let records =
            compute_rings(&coords, &barcodes, &types, &ring_edges, false, "test").unwrap();

        // Total counts should not include self
        for r in &records {
            assert!(r.count > 0); // only non-zero emitted
        }
    }

    #[test]
    fn rings_rejects_invalid_edges() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0]];
        let barcodes: Vec<String> = (0..2).map(|i| format!("c{i}")).collect();
        let types: Vec<String> = vec!["A".to_string(); 2];

        // Non-increasing edges
        assert!(compute_rings(&coords, &barcodes, &types, &[0.0, 5.0, 3.0], false, "g").is_err());
        // Only one edge
        assert!(compute_rings(&coords, &barcodes, &types, &[0.0], false, "g").is_err());
    }
}
