use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use serde::Serialize;
use spatialrs_core::{
    composition::compute_composition,
    interactions::count_interactions,
    neighbors::{knn_graph, radius_graph, EdgeRecord},
};
use spatialrs_io::{read_h5ad, AnnData};
use std::path::PathBuf;

// ─── CLI definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "spatialrs", about = "Spatial transcriptomics graph tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute radius-based neighbour graph
    Radius {
        input: PathBuf,
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compute k-nearest-neighbour graph
    Knn {
        input: PathBuf,
        #[arg(long)]
        k: usize,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Count cell-type pair interactions within a radius
    Interactions {
        input: PathBuf,
        #[arg(long)]
        cell_type: String,
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compute per-cell neighbourhood composition
    Composition {
        input: PathBuf,
        #[arg(long)]
        cell_type: String,
        #[arg(long)]
        radius: f64,
        #[arg(long, default_value = "sample")]
        groupby: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

// ─── entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Radius { input, radius, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby])?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<EdgeRecord> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    radius_graph(&coords, &barcodes, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Knn { input, k, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby])?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<EdgeRecord> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    knn_graph(&coords, &barcodes, k, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Interactions { input, cell_type, radius, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby, &cell_type])?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<_> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let types = extract_strings_subset(&adata, &cell_type, indices);
                    count_interactions(&coords, &barcodes, &types, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }

        Command::Composition { input, cell_type, radius, groupby, output } => {
            let adata = read_h5ad(&input, &[&groupby, &cell_type])?;
            let groups = partition_by_group(&adata, &groupby)?;

            let records: Vec<_> = groups
                .par_iter()
                .map(|(label, indices)| {
                    let coords = extract_coords_subset(&adata, indices);
                    let barcodes = extract_barcodes_subset(&adata, indices);
                    let types = extract_strings_subset(&adata, &cell_type, indices);
                    compute_composition(&coords, &barcodes, &types, radius, label)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            write_csv(&records, output.as_deref())?;
        }
    }

    Ok(())
}

// ─── shared helpers ───────────────────────────────────────────────────────────

/// Group cell indices by the value of the specified obs column.
fn partition_by_group(adata: &AnnData, groupby: &str) -> Result<Vec<(String, Vec<usize>)>> {
    let col = adata
        .obs
        .get(groupby)
        .with_context(|| format!("obs column '{groupby}' not loaded"))?;

    let mut map: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, label) in col.iter().enumerate() {
        map.entry(label.clone()).or_default().push(i);
    }

    let mut groups: Vec<(String, Vec<usize>)> = map.into_iter().collect();
    groups.sort_by(|a, b| a.0.cmp(&b.0));   // deterministic ordering
    Ok(groups)
}

fn extract_coords_subset(adata: &AnnData, indices: &[usize]) -> Vec<[f64; 2]> {
    indices
        .iter()
        .map(|&i| [adata.coordinates[[i, 0]], adata.coordinates[[i, 1]]])
        .collect()
}

fn extract_barcodes_subset(adata: &AnnData, indices: &[usize]) -> Vec<String> {
    indices.iter().map(|&i| adata.obs_names[i].clone()).collect()
}

fn extract_strings_subset(adata: &AnnData, col: &str, indices: &[usize]) -> Vec<String> {
    let values = &adata.obs[col];
    indices.iter().map(|&i| values[i].clone()).collect()
}

fn write_csv<T: Serialize>(records: &[T], output: Option<&std::path::Path>) -> Result<()> {
    let writer: Box<dyn std::io::Write> = match output {
        Some(path) => Box::new(
            std::fs::File::create(path)
                .with_context(|| format!("cannot create {:?}", path))?,
        ),
        None => Box::new(std::io::stdout()),
    };
    let mut wtr = csv::Writer::from_writer(writer);

    for record in records {
        wtr.serialize(record).context("serializing record")?;
    }
    wtr.flush().context("flushing CSV writer")?;
    Ok(())
}
