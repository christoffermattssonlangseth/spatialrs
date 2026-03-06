# spatialrs

A fast command-line toolkit for spatial transcriptomics analysis, written in Rust.

Reads `.h5ad` files and provides graph construction, interaction counting, composition profiling, NMF factorization, and spatial aggregation — all parallelized with Rayon.

---

## Crates

| Crate | Role |
|---|---|
| `spatialrs-io` | HDF5 reader: obs/var names, spatial coords, obsm embeddings, expression matrix X |
| `spatialrs-core` | Algorithms: neighbors, interactions, composition, NMF, aggregation |
| `spatialrs-cli` | Binary: CLI wiring with Clap |

---

## Building

```bash
cargo build --release
# binary at target/release/spatialrs
```

Requires a system HDF5 library (the `hdf5-metno` crate links against it).

---

## Subcommands

### `radius` — radius-based neighbour graph

```bash
spatialrs radius data.h5ad --radius 50 --groupby sample --output edges.csv
```

Output columns: `cell_i, cell_j, distance, group`
Edges are bidirectional (both i→j and j→i).

---

### `knn` — k-nearest-neighbour graph

```bash
spatialrs knn data.h5ad --k 6 --groupby sample --output edges.csv
```

Output columns: `cell_i, cell_j, distance, group`

---

### `interactions` — cell-type pair interaction counts

```bash
spatialrs interactions data.h5ad \
    --cell-type cell_type \
    --radius 50 \
    --groupby sample \
    --output interactions.csv
```

Counts co-localised cell-type pairs within `radius`. Each pair is counted once (canonical alphabetical order). Output columns: `group, cell_type_a, cell_type_b, count`

---

### `composition` — per-cell neighbourhood composition

```bash
spatialrs composition data.h5ad \
    --cell-type cell_type \
    --radius 50 \
    --groupby sample \
    --output composition.csv
```

For each cell, computes the fraction of each cell type among its neighbours within `radius`. Isolated cells are omitted. Output columns: `cell_i, cell_type, fraction, group`

---

### `nmf` — non-negative matrix factorization

Factorizes the gene expression matrix X (cells × genes) into W (cell factors) and H (gene loadings) using multiplicative update rules (Lee & Seung 2001).

```bash
# All cells as one group
spatialrs nmf data.h5ad \
    --n-components 20 \
    --output-w w_factors.csv \
    --output-h h_loadings.csv

# Per-sample factorization
spatialrs nmf data.h5ad \
    --n-components 20 \
    --groupby sample \
    --output-w w_factors.csv \
    --output-h h_loadings.csv
```

| Flag | Default | Description |
|---|---|---|
| `--n-components` | 10 | Number of NMF components |
| `--max-iter` | 200 | Maximum multiplicative update iterations |
| `--tol` | 1e-4 | Convergence tolerance on Frobenius error change |
| `--seed` | 42 | RNG seed for reproducible initialisation |
| `--groupby` | *(none)* | Optional obs column to partition cells before factorizing |
| `--output-w` | stdout | W matrix CSV (long format: `cell_i, component, weight, group`) |
| `--output-h` | *(skip)* | H matrix CSV (long format: `gene, component, loading, group`) |

Convergence status is printed to stderr per group.

---

### `aggregate` — distance-weighted spatial aggregation

For each cell, computes a weighted average of its neighbours' embedding vectors (e.g. `X_scVI`).

```bash
spatialrs aggregate data.h5ad \
    --embedding X_scVI \
    --radius 30 \
    --weighting gaussian \
    --sigma 15 \
    --groupby sample \
    --output agg.csv
```

**Graph mode** (pick one):

| Flag | Description |
|---|---|
| `--radius <f64>` | All neighbours within this distance |
| `--k <usize>` | k nearest neighbours |

**Weighting modes:**

| `--weighting` | Additional flag | Formula |
|---|---|---|
| `uniform` | — | w = 1 |
| `gaussian` | `--sigma <f64>` | w = exp(−d² / 2σ²) |
| `exponential` | `--decay <f64>` | w = exp(−λd) |
| `inverse-distance` | `--epsilon <f64>` (default 1e-9) | w = 1 / (d + ε) |

Output columns: `cell_i, dim, value, group` (long format, one row per cell × embedding dimension).
Cells with no neighbours within the search radius emit zeros.

---

### `gmm` — Gaussian Mixture Model niche detection

Fits a GMM to an embedding to identify spatial compartments / niches.
The typical pipeline is:

```bash
# 1. Build distance-weighted aggregated embedding per cell
spatialrs aggregate data.h5ad \
    --embedding X_scVI --radius 30 --weighting gaussian --sigma 15 \
    --groupby sample --output agg.csv

# 2. Cluster the aggregated embeddings into niches
spatialrs gmm data.h5ad --agg agg.csv \
    -k 10 --covariance diagonal --groupby sample \
    --output niches.csv --output-probs probs.csv
```

You can also feed the GMM raw obsm embeddings or NMF factors directly:

```bash
spatialrs gmm data.h5ad --embedding X_scVI -k 10 --groupby sample --output niches.csv
spatialrs gmm data.h5ad --nmf-w w_factors.csv -k 10 --groupby sample --output niches.csv
```

**Embedding source** (exactly one required):

| Flag | Description |
|---|---|
| `--embedding <obsm_key>` | Load directly from obsm |
| `--nmf-w <path>` | Load from NMF W factors CSV |
| `--agg <path>` | Load from aggregation CSV (output of `spatialrs aggregate`) |

**GMM parameters:**

| Flag | Default | Description |
|---|---|---|
| `-k / --k` | *(required)* | Number of mixture components (niches) |
| `--covariance` | `diagonal` | `diagonal` or `spherical` covariance |
| `--max-iter` | 200 | Maximum EM iterations |
| `--tol` | 1e-6 | Convergence tolerance on log-likelihood change |
| `--seed` | 42 | RNG seed for K-means++ initialisation |
| `--reg-covar` | 1e-6 | Regularisation added to variance to prevent singularity |
| `--groupby` | `sample` | Obs column; GMM is fit independently per group |

**Outputs:**
- `--output` — hard assignments: `cell_i, niche, group`
- `--output-probs` *(optional)* — soft responsibilities: `cell_i, component, probability, group`

Convergence status (log-likelihood, iterations) is printed to stderr per group.

---

## Input format

Standard `.h5ad` files written by AnnData (Python).

- **Spatial coordinates** read from `obsm/spatial` or `obsm/X_spatial` (first two columns used).
- **Obs columns** read from `obs/{col}` — supports both categorical (codes/categories) and plain string datasets.
- **obsm embeddings** read from `obsm/{key}` — f32 or f64, all columns.
- **Expression matrix X** — supports sparse CSR groups (`X/data`, `X/indices`, `X/indptr`) and dense datasets.

---

## Design notes

- Groups (samples) are processed in parallel with Rayon; spatial indexing uses an R* tree (`rstar`).
- NMF element-wise updates use `ndarray::Zip::par_for_each` across rayon threads.
- All CSV outputs are long format to be compatible with any number of components or embedding dimensions.
- Cells without neighbours in `aggregate` are kept in the output (zeros) to preserve a 1-to-1 join with the input.
