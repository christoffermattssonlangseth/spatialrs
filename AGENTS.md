# AGENTS.md — spatialrs

Guidelines for AI agents working in this repository.

---

## Repository structure

```
spatialrs/
├── scripts/                      Local runner scripts for full pipelines / datasets
│   ├── end-to-end-pipeline.sh
│   ├── runspatialrs-balo.sh
│   ├── runspatialrs-hm-sc.sh
│   └── runspatialrs-rrmap.sh
├── notebooks/                    Analysis notebooks that join CSV outputs back to AnnData
│   ├── explore_niches-*.ipynb
│   └── map_spatialrs_outputs_to_h5ad.ipynb
├── spatialrs-io/src/lib.rs       HDF5 reader (AnnData struct + read_h5ad)
├── spatialrs-core/src/
│   ├── lib.rs                    Module declarations
│   ├── neighbors.rs              Radius graph, kNN graph (rstar + rayon)
│   ├── interactions.rs           Cell-type pair interaction counts + permutation test
│   ├── composition.rs            Per-cell neighbourhood composition
│   ├── diff_composition.rs       Differential composition test (MWU + BH correction)
│   ├── nmf.rs                    NMF (multiplicative updates, ndarray + rayon)
│   ├── aggregation.rs            Distance-weighted spatial aggregation
│   ├── gmm.rs                    Gaussian mixture niche detection
│   ├── autocorr.rs               Global Moran's I + permutation p-values + Geary's C + LISA + bivariate Moran's I
│   ├── local_cor.rs              Per-cell local Pearson correlation within a radius neighbourhood
│   ├── rings.rs                  Concentric-ring neighbourhood composition
│   ├── transitions.rs            Niche spatial co-occurrence matrix + permutation test
│   ├── ripley.rs                 Ripley's K / L + cross-K / cross-L
│   ├── diff_niches.rs            Sample-level niche abundance differential test
│   └── markers.rs                Pooled niche marker detection (Wilcoxon)
Note: `compute_graph_stats` lives in `neighbors.rs` (no separate module needed).
└── spatialrs-cli/src/main.rs     Clap CLI; one match arm per subcommand
```

Generated outputs such as project-local `.h5ad` exports, CSVs, and the Rust `target/` directory are not part of the source tree even if they exist locally.

---

## Build & test

```bash
cargo build --release   # compile
cargo test              # run all unit tests
```

Both commands must succeed with zero errors and zero warnings before committing.

---

## Adding a new subcommand

1. **Core logic** — add a new module under `spatialrs-core/src/`. Export it from `lib.rs`.
2. **IO** — if new h5ad fields are needed, extend `AnnData` in `spatialrs-io/src/lib.rs` and update `read_h5ad` (signature: `path, obs_cols, obsm_keys, load_expression, load_sparse, var_filter, layer`). Update all existing call sites.
3. **CLI** — add a variant to the `Command` enum in `spatialrs-cli/src/main.rs`, a match arm in `main()`, and any helper parsing functions. Pass `false, false, None, None` for unused `load_expression`/`load_sparse`/filter/layer arguments on existing subcommands.
4. **Cargo.toml** — add new dependencies at workspace level (`Cargo.toml`) and reference them from the relevant member crate's `Cargo.toml`.
5. **Docs** — update `README.md` with the new subcommand section and `AGENTS.md` with the new module in the repository structure.

---

## Typical spatial niche pipeline

```bash
spatialrs nmf data.h5ad --n-components 20 --groupby sample --output-w w.csv
spatialrs aggregate data.h5ad --nmf-w w.csv --radius 30 --weighting gaussian --sigma 15 --groupby sample --output agg.csv
spatialrs gmm data.h5ad --agg agg.csv -k 10 --covariance diagonal --groupby sample --output niches.csv
```

Each step is independent and outputs a flat CSV, making them easy to inspect or substitute.

To choose k, use `gmm-sweep` first:

```bash
spatialrs gmm-sweep data.h5ad --agg agg.csv --k-min 2 --k-max 20 --output sweep.csv
```

To choose the NMF rank, use `nmf-sweep` first:

```bash
spatialrs nmf-sweep data.h5ad --k-min 2 --k-max 20 --output nmf_sweep.csv
```

Downstream analysis after niche detection:

```bash
# Marker genes per niche
spatialrs markers data.h5ad --niche-csv niches.csv --output markers.csv

# Spatial structure of niches (co-occurrence + permutation enrichment test)
spatialrs transitions data.h5ad --niche-csv niches.csv --radius 50 --groupby sample \
    --output transitions.csv --output-stats transition_stats.csv

# Per-sample niche summary for QC / external tools
spatialrs summarize niches.csv --output niche_summary.csv

# Differential niche abundance between conditions (sample-level test)
spatialrs diff-niches data.h5ad --niche-csv niches.csv \
    --condition condition --groupby sample --group-a tumor --group-b normal \
    --output diff_niches.csv

# Spatial autocorrelation of NMF components
spatialrs morans data.h5ad --nmf-w w.csv --radius 50 --groupby sample --output morans.csv \
    --n-permutations 999 --output-perm morans_perm.csv
spatialrs geary data.h5ad --nmf-w w.csv --radius 50 --groupby sample --output geary.csv
spatialrs lisa data.h5ad --nmf-w w.csv --radius 50 --groupby sample --output lisa.csv
spatialrs bivariate-morans data.h5ad --nmf-w w.csv --radius 50 --groupby sample --output biv_morans.csv

# QC: choose radius by inspecting per-cell neighbour counts
spatialrs graph-stats data.h5ad --radius 50 --groupby sample --output graph_stats.csv

# Spatially variable genes
spatialrs svg data.h5ad --radius 50 --groupby sample --var-filter highly_variable --output svg.csv

# Cell-type spatial clustering
spatialrs ripley data.h5ad --cell-type cell_type --target-type "Tumor cell" \
    --r-min 10 --r-max 200 --groupby sample --output ripley.csv
spatialrs cross-ripley data.h5ad --cell-type cell_type --type-a "T cell" --type-b "Tumor cell" \
    --r-min 10 --r-max 200 --groupby sample --output cross_ripley.csv
spatialrs ripley-envelope data.h5ad --cell-type cell_type --target-type "Tumor cell" \
    --r-min 10 --r-max 200 --groupby sample --output ripley_env.csv
spatialrs cross-ripley-envelope data.h5ad --cell-type cell_type --type-a "T cell" --type-b "Tumor cell" \
    --r-min 10 --r-max 200 --groupby sample --output cross_ripley_env.csv

# Neighbourhood composition and entropy
spatialrs composition data.h5ad --cell-type cell_type --radius 50 --groupby sample \
    --output comp.csv --output-entropy entropy.csv
spatialrs rings data.h5ad --cell-type cell_type --rings 0,25,50,100 --groupby sample \
    --output rings.csv

# Differential neighbourhood composition between conditions (cell level)
spatialrs diff-composition data.h5ad --composition-csv comp.csv \
    --condition condition --group-a tumor --group-b normal --output diff.csv

# Per-cell local correlation between two features
spatialrs local-cor data.h5ad --feature-a X_scVI:0 --feature-b X_scVI:1 \
    --radius 50 --groupby sample --output local_cor.csv
```

Local helper scripts for these workflows now live under `scripts/`. If you adjust CLI flags or expected outputs, update the corresponding script(s) as part of the same change.

The notebook [`notebooks/map_spatialrs_outputs_to_h5ad.ipynb`](/Users/christoffer/work/karolinska/development/spatialrs/notebooks/map_spatialrs_outputs_to_h5ad.ipynb) is the current reference for mapping `spatialrs` outputs back into `AnnData` (`obs`, `obsm`, `varm`, `obsp`, `uns`) and for downstream niche/marker analysis.

---

## Key patterns

### Reading h5ad data
Always call `read_h5ad` with only the obs columns and obsm keys you actually need.
Passing `load_expression: true` reads the full dense X matrix — only do this for NMF or similar.
Use `load_sparse: true` only for sparse NMF paths, and keep `load_expression` and `load_sparse` mutually exclusive.

### Parallelism
Groups (samples) are processed in parallel with `groups.par_iter()` in the CLI. Core functions are single-group and must be safe to call from rayon threads (no `Mutex`, no global state).

Within `compute_local_morans_i` (LISA), parallelism is over features using `par_iter().flat_map_iter()`. The inner closure must be `Fn` (not `FnOnce`): do not `move` the adjacency list into the inner map; instead collect into a `Vec<_>` inside the outer closure so borrows are scoped.

### Output format
All outputs are long-format CSV via `write_csv`. Add a `group` field to every record struct so outputs from multi-group runs can be filtered downstream.
Exception: `markers` is intentionally pooled across all cells, so `MarkerRecord` does not include `group`.

### Spatial indexing
Use `rstar::RTree` with the `IndexedPoint` pattern (see `neighbors.rs` or `aggregation.rs`). Build the tree with `RTree::bulk_load`. Query radius neighbours with `locate_within_distance(point, r²)` (note: squared radius). Query kNN with `nearest_neighbor_iter_with_distance_2`.

Each module that needs spatial indexing defines its own private `IndexedPoint` struct — `neighbors.rs` does not re-export it.

`radius_graph_index_pairs` in `neighbors.rs` is `pub(crate)` and used by `interactions.rs` and `transitions.rs` for efficient permutation/co-occurrence counting without constructing full `EdgeRecord`s.

### NMF
Uses `ndarray::Zip::par_for_each` for element-wise updates (requires `ndarray` rayon feature). Matrix multiplications are via `Array2::dot`. Convergence is checked every 10 iterations using Frobenius norm. W shape: `(n_obs, k)`; H shape: `(k, n_var)`.

`NmfResult` carries `component_variances` (per-component ‖W[:,k]‖₂ × ‖H[k,:]‖₂ normalised to sum to 1) and `error_trajectory` (Vec of (iteration, error) checkpoints). Both are populated by `run_nmf` and `run_nmf_sparse`.

`nmf-sweep` lives in the CLI and loops over `k` values sequentially, outputting `NmfSweepRecord` rows; no separate core module is needed beyond `run_nmf_sweep` / `run_nmf_sweep_sparse`.

### GMM
EM algorithm in `spatialrs-core/src/gmm.rs`. E-step is parallelized over cells with rayon. Variances are always stored as `Array2<f64>` of shape `(K, D)` — for spherical covariance, all columns in a row hold the same scalar. K-means++ initialisation. Convergence checked every iteration on log-likelihood change. Outputs hard labels (`labels: Vec<usize>`) and soft responsibilities (`Array2<f64>` N×K), plus `bic` and `aic`.

`gmm-sweep` in the CLI loops over k values sequentially and outputs `ModelStatsRecord` for each — no new core code needed.

### Markers
`markers` is a pooled one-vs-rest test over the full expression matrix. It consumes niche labels from a CSV, aligns them to `obs_names`, and does not split work by sample.

### Transitions
`transitions.rs` uses `radius_graph_index_pairs` to build upper-triangle edge pairs, then counts niche label co-occurrences into a `K×K` triangular matrix. Returns all upper-triangle entries (including diagonal). `n_niches` is derived from `max(niche_label) + 1` in the CLI.

`permute_transitions` shuffles niche labels `n_perms` times (seeded per-permutation), reuses the same edge list, and returns z-scores and empirical p-values for each niche pair. The `pair_to_flat_idx` helper uses the formula `a*(2*n-a+1)/2 + (b-a)` to avoid usize underflow when `a=0`.

### Ripley's K/L and Cross-K/L
`ripley.rs` provides two functions:
- `compute_ripley` — single-type K(r)/L(r), parallelized over radii.
- `compute_cross_ripley` — bivariate K_AB(r)/L_AB(r); bounding box covers both type A and B cells. When `type_a == type_b` self-pairs are excluded via a `same_type` flag inside the query filter.

Both use bounding-box area clamped to 1.0 as the CSR reference area.

`ripley.rs` also provides `ripley_envelope` and `cross_ripley_envelope`, which generate simulation envelopes for the corresponding statistics in the CLI `ripley-envelope` and `cross-ripley-envelope` subcommands.

### Geary's C
`compute_gearys_c` in `autocorr.rs` follows the same graph-build pattern as `compute_morans_i`. Formula: `C = ((n-1)/S₀) × Σ_edge (zᵢ-zⱼ)² / Σ zᵢ²`. Variance: `[(2S₁+S₂)(n-1) - 4S₀²] / [2(n+1)S₀²]`. For binary symmetric weights: S₁=2S₀, S₂=4Σdegᵢ². C < 1 = positive autocorrelation; C > 1 = negative (opposite direction to Moran's I z-score).

### Moran's I permutation test
`compute_morans_i_perm` builds the graph once, computes observed I, then generates `n_perms` label-shuffled nulls in parallel (seeded per-permutation). Two-tailed: counts |I_perm| ≥ |I_obs|. Returns `(n_exceeding+1)/(n_perms+1)` per feature. The inner `moran_i` closure captures edge_pairs and values by immutable reference — this is valid in Rayon since both are `Sync`.

### Graph stats
`compute_graph_stats` in `neighbors.rs` queries each cell's R*-tree neighbourhood and counts neighbors excluding self. No separate module needed. Used by the `graph-stats` CLI subcommand.

### SVG (spatially variable genes)
The `Svg` CLI subcommand loads the expression matrix (`load_expression: true`), maps it from `f32` to `f64` with `.mapv(|v| v as f64)`, and calls `compute_morans_i` per group. Output reuses `MoranRecord`; feature names come from `adata.var_names`. Supports `--var-filter` and `--layer` like `nmf`.

### Bivariate Moran's I
`compute_bivariate_morans_i` in `autocorr.rs` computes the bivariate statistic for all unique feature pairs (f_a < f_b). The null expectation is E[I_AB] = 0; the variance uses only the graph topology (sum of squared weights squared). The z-score is topology-only (does not depend on values).

### Differential composition
`diff_composition.rs` accepts a flat list of `(barcode, cell_type, fraction)` tuples plus a `condition_map`. It partitions fractions by condition and runs per-cell-type Mann–Whitney U tests (same helper functions as `markers.rs`). BH correction applied across all cell types.

### Differential niches
`diff_niches.rs` takes per-sample niche fraction vectors (one `Vec<f64>` of length `n_niches` per sample). In the CLI, these are built by grouping the niche CSV by `group` column (= sample), counting cells per niche, and normalising by sample total. Condition membership is looked up from the h5ad `obs` `--condition` column. MWU with BH correction is applied across niches.

### Spatial entropy
`compute_entropy` in `composition.rs` accepts a slice of `CompositionRecord` and groups by `(cell_i, group)`, computing H = −Σ pₖ log₂(pₖ). Called in the `Composition` CLI arm when `--output-entropy` is provided, operating on the already-collected composition records.

### Summarize
`Summarize` in the CLI reads a niche CSV (no h5ad), groups by `group` column, counts cells per niche, and emits `NicheSummaryRecord`. Sorted by (group, niche) for stable output.

### Local correlation
`compute_local_cor` in `local_cor.rs` computes a per-cell Pearson correlation between two features inside each cell's radius neighbourhood. The neighbourhood includes the focal cell itself. `local_r` is `NaN` when fewer than 3 cells are present locally or when either feature has zero local variance.

In the CLI, features may come from three sources:
- bare gene name, e.g. `GAPDH`
- obsm column spec, e.g. `X_pca:3`
- NMF component spec with `--nmf-w`, e.g. `nmf:0`

### Rings
`compute_rings` in `rings.rs` computes per-cell neighbourhood composition over concentric distance bins. `ring_edges` must be strictly increasing and define half-open intervals `[inner, outer)`. With `--include-zeros`, the CLI emits zero-count rows for all observed cell types in every ring; otherwise zero-count combinations are omitted.

### Embedding sources in `aggregate`, `gmm`, `gmm-sweep`, `morans`, `geary`, `lisa`, and `bivariate-morans`
These subcommands accept three embedding sources via CLI flags:
- `--embedding <key>` → load from obsm in the h5ad
- `--nmf-w <path>` → pivot NMF W CSV (long format) into dense matrix via `read_nmf_w_embedding`
- `--agg <path>` → pivot aggregation CSV (long format) into dense matrix via `read_agg_embedding`
Both helper functions are in `spatialrs-cli/src/main.rs` and validate against obs_names from the h5ad.

---

## Constraints

- Do not add dependencies without a clear reason. Prefer the existing stack: `ndarray`, `rayon`, `rstar`, `hdf5-metno`, `clap`, `csv`, `serde`, `anyhow`.
- Do not use `unsafe` code.
- Do not use `unwrap()` or `expect()` in library code (`spatialrs-core`, `spatialrs-io`). Propagate errors with `?` and `anyhow::Context`.
- All CSV record structs must derive `serde::Serialize` and include a `group: String` field, except pooled `MarkerRecord`.
- The `read_h5ad` signature is fixed: `(path, obs_cols, obsm_keys, load_expression, load_sparse, var_filter, layer)`. Do not change it without updating all call sites.
- Do not commit generated artifacts such as `target/`, `.DS_Store`, project-local `.h5ad` files, or ad hoc output CSVs produced by running the CLI.

---

## Common pitfalls

- **Generated files vs source files**: repo-cleanup has already removed tracked `target/` output and Finder metadata. Keep it that way. If a change only affects generated files, do not stage them.
- **`ndarray::Axis`** is not re-exported by spatialrs-core. If you need it in the CLI, add `ndarray` as a direct dependency of `spatialrs-cli`.
- **Sparse X matrices** in h5ad may use `i32`, `i64`, `u32`, or `u64` for `indptr`/`indices`. Use `read_usize_vec` in `spatialrs-io` to handle all variants.
- **obs column encoding**: h5ad writes categorical columns as `codes + categories` groups, not flat arrays. `read_obs_column` handles both; do not bypass it.
- **R* tree distances**: `locate_within_distance` takes the squared radius; `distance_2` returns squared distance. Apply `.sqrt()` when you need actual distance.
- **`flat_map_iter` closures must be `Fn`**: any variables captured from the outer scope that are not `Copy` must be accessed by reference (not moved). Collect inner results into a `Vec<_>` before returning so temporary borrows are released within the single closure invocation.
- **Ripley area floor**: the bounding-box area for target cells is clamped to 1.0 to prevent division-by-zero when all target cells are collinear (e.g. in unit tests).
- **`pair_to_flat_idx` with `a=0`**: the formula `a*(2*n-a+1)/2 + (b-a)` avoids usize underflow. Do NOT write `a*n - a*(a-1)/2` — `a*(a-1)` for `a=0usize` evaluates `0-1` which panics with subtract-with-overflow in debug mode.
- **Cross-Ripley test geometry**: for `L_AB(r) > 0`, the bounding box area must exceed `π·r²`. In tests, put some type-A cells far apart to inflate the bounding box, then cluster type-B cells near one type-A cell so pair counts are high relative to the large area.
- **Moran's I permutation degenerate case**: with very few cells (≤4) and binary values, ALL label permutations may give |I|=1 (maximum), causing p_value_perm=1 regardless of observed I. Use ≥8 cells and ≥2 edges per cluster in permutation tests.
- **SVG vs morans**: `svg` loads the expression matrix (`load_expression: true`); `morans` loads an obsm embedding or NMF W CSV. Both call `compute_morans_i` internally.
- **Geary's C sign**: z_score < 0 means positive spatial autocorrelation (C < 1), opposite sign convention to Moran's I where z_score > 0 means positive. Keep this in mind when interpreting combined results.
- **`diff-niches` vs `diff-composition`**: `diff-niches` operates at the **sample level** (one observation = one sample's niche fraction); `diff-composition` operates at the **cell level** (one observation = one cell's neighbourhood fraction). Use `diff-niches` for multi-sample comparisons; use `diff-composition` for single-sample conditions.
- **Notebook execution state**: when editing `.ipynb` files, avoid leaving misleading cached outputs that no longer match the source cells. Prefer clean JSON with source-of-truth in the cell bodies.
- **`local-cor` feature parsing**: specs containing `:` are interpreted as `obsm_key:column`, except `nmf:<component>` which requires `--nmf-w`. Bare strings are treated as gene names and therefore require loading the expression matrix.
- **Ring boundaries**: `--rings` must define at least two edges and they must be strictly increasing. Distances are binned into half-open intervals `[edge_k, edge_{k+1})`, so points exactly on the upper boundary fall into the next ring.
