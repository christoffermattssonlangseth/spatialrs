use crate::neighbors::radius_graph_dedup;
use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct InteractionRecord {
    pub group: String,
    pub cell_type_a: String,
    pub cell_type_b: String,
    pub count: usize,
}

/// Count cell-type pair interactions within `radius`.
/// Each pair is counted once (canonical ordering: a ≤ b alphabetically).
pub fn count_interactions(
    coords: &[[f64; 2]],
    barcodes: &[String],
    cell_types: &[String],
    radius: f64,
    group: &str,
) -> Result<Vec<InteractionRecord>> {
    let edges = radius_graph_dedup(coords, barcodes, radius, group)?;

    // Build barcode → cell_type lookup
    let type_map: HashMap<&str, &str> = barcodes
        .iter()
        .zip(cell_types.iter())
        .map(|(b, t)| (b.as_str(), t.as_str()))
        .collect();

    let mut counts: HashMap<(String, String), usize> = HashMap::new();

    for edge in &edges {
        let ta = type_map[edge.cell_i.as_str()];
        let tb = type_map[edge.cell_j.as_str()];
        // Canonical order
        let key = if ta <= tb {
            (ta.to_string(), tb.to_string())
        } else {
            (tb.to_string(), ta.to_string())
        };
        *counts.entry(key).or_insert(0) += 1;
    }

    let records = counts
        .into_iter()
        .map(|((a, b), count)| InteractionRecord {
            group: group.to_string(),
            cell_type_a: a,
            cell_type_b: b,
            count,
        })
        .collect();

    Ok(records)
}
