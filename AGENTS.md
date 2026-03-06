# AGENTS.md — spatialrs

Guidelines for AI agents working in this repository.

---

## Repository structure

```
spatialrs/
├── spatialrs-io/src/lib.rs       HDF5 reader (AnnData struct + read_h5ad)
├── spatialrs-core/src/
│   ├── lib.rs                    Module declarations
│   ├── neighbors.rs              Radius graph, kNN graph (rstar + rayon)
│   ├── interactions.rs           Cell-type pair interaction counts
│   ├── composition.rs            Per-cell neighbourhood composition
│   ├── nmf.rs                    NMF (multiplicative updates, ndarray + rayon)
│   └── aggregation.rs            Distance-weighted spatial aggregation
└── spatialrs-cli/src/main.rs     Clap CLI; one match arm per subcommand
```

---

## Build & test

```bash
cargo build --release   # compile
cargo test              # run all tests (currently compile-check only)
```

Both commands must succeed with zero errors and zero warnings before committing.

---

## Adding a new subcommand

1. **Core logic** — add a new module under `spatialrs-core/src/`. Export it from `lib.rs`.
2. **IO** — if new h5ad fields are needed, extend `AnnData` in `spatialrs-io/src/lib.rs` and update `read_h5ad` (signature: `path, obs_cols, obsm_keys, load_expression`). Update all existing call sites.
3. **CLI** — add a variant to the `Command` enum in `spatialrs-cli/src/main.rs`, a match arm in `main()`, and any helper parsing functions. Pass `&[], false` for unused `obsm_keys`/`load_expression` on existing subcommands.
4. **Cargo.toml** — add new dependencies at workspace level (`Cargo.toml`) and reference them from the relevant member crate's `Cargo.toml`.

---

## Key patterns

### Reading h5ad data
Always call `read_h5ad` with only the obs columns and obsm keys you actually need.
Passing `load_expression: true` reads the full dense X matrix — only do this for NMF or similar.

### Parallelism
Groups (samples) are processed in parallel with `groups.par_iter()` in the CLI. Core functions are single-group and must be safe to call from rayon threads (no `Mutex`, no global state).

### Output format
All outputs are long-format CSV via `write_csv`. Add a `group` field to every record struct so outputs from multi-group runs can be filtered downstream.

### Spatial indexing
Use `rstar::RTree` with the `IndexedPoint` pattern (see `neighbors.rs` or `aggregation.rs`). Build the tree with `RTree::bulk_load`. Query radius neighbours with `locate_within_distance(point, r²)` (note: squared radius). Query kNN with `nearest_neighbor_iter_with_distance_2`.

### NMF
Uses `ndarray::Zip::par_for_each` for element-wise updates (requires `ndarray` rayon feature). Matrix multiplications are via `Array2::dot`. Convergence is checked every 10 iterations using Frobenius norm. W shape: `(n_obs, k)`; H shape: `(k, n_var)`.

---

## Constraints

- Do not add dependencies without a clear reason. Prefer the existing stack: `ndarray`, `rayon`, `rstar`, `hdf5-metno`, `clap`, `csv`, `serde`, `anyhow`.
- Do not use `unsafe` code.
- Do not use `unwrap()` or `expect()` in library code (`spatialrs-core`, `spatialrs-io`). Propagate errors with `?` and `anyhow::Context`.
- All CSV record structs must derive `serde::Serialize` and include a `group: String` field.
- The `read_h5ad` signature is fixed: `(path, obs_cols, obsm_keys, load_expression)`. Do not change it without updating all call sites.

---

## Common pitfalls

- **`ndarray::Axis`** is not re-exported by spatialrs-core. If you need it in the CLI, add `ndarray` as a direct dependency of `spatialrs-cli`.
- **Sparse X matrices** in h5ad may use `i32`, `i64`, `u32`, or `u64` for `indptr`/`indices`. Use `read_usize_vec` in `spatialrs-io` to handle all variants.
- **obs column encoding**: h5ad writes categorical columns as `codes + categories` groups, not flat arrays. `read_obs_column` handles both; do not bypass it.
- **R* tree distances**: `locate_within_distance` takes the squared radius; `distance_2` returns squared distance. Apply `.sqrt()` when you need actual distance.
