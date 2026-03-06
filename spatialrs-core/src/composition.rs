use crate::neighbors::radius_graph;
use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct CompositionRecord {
    pub cell_i: String,
    pub cell_type: String,
    pub fraction: f64,
    pub group: String,
}

/// Compute the per-cell neighbourhood composition within `radius`.
/// For each cell, counts the cell types among its neighbours and returns
/// one row per (cell, neighbour_type) with the fraction of that type.
pub fn compute_composition(
    coords: &[[f64; 2]],
    barcodes: &[String],
    cell_types: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<CompositionRecord>> {
    // Bidirectional edges so iterating by cell_i gives all neighbours
    let edges = radius_graph(coords, barcodes, radius, group)?;

    let type_map: HashMap<&str, &str> = barcodes
        .iter()
        .zip(cell_types.iter())
        .map(|(b, t)| (b.as_str(), t.as_str()))
        .collect();

    // Build neighbour_map: cell_barcode → list of neighbour cell_types
    let mut neighbour_map: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &edges {
        let nb_type = type_map[edge.cell_j.as_str()];
        neighbour_map
            .entry(edge.cell_i.as_str())
            .or_default()
            .push(nb_type);
    }

    let mut records = Vec::new();

    for barcode in barcodes {
        let neighbours = match neighbour_map.get(barcode.as_str()) {
            Some(nb) => nb,
            None => continue,   // isolated cell — skip
        };

        let total = neighbours.len() as f64;
        let mut type_counts: HashMap<&str, usize> = HashMap::new();
        for &t in neighbours {
            *type_counts.entry(t).or_insert(0) += 1;
        }

        for (ct, cnt) in type_counts {
            records.push(CompositionRecord {
                cell_i: barcode.clone(),
                cell_type: ct.to_string(),
                fraction: cnt as f64 / total,
                group: group.to_string(),
            });
        }
    }

    Ok(records)
}
