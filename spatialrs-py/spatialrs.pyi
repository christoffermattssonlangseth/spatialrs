"""
spatialrs — fast spatial transcriptomics analysis (Rust core, Python bindings).

All functions that return tabular data yield ``list[dict]`` which converts
directly to a DataFrame via ``pd.DataFrame(result)``.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import numpy.typing as npt

# ─── type aliases ─────────────────────────────────────────────────────────────

Coords   = npt.NDArray[np.float64]   # shape (N, 2)
Matrix   = npt.NDArray[np.float64]   # shape (N, D)
Vector   = npt.NDArray[np.float64]   # shape (N,)
Row      = dict[str, object]
Table    = list[Row]

# ─── graph construction ───────────────────────────────────────────────────────

def radius_graph(
    coords: Coords,
    barcodes: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Build a radius graph.

    Returns
    -------
    list[dict]  —  keys: cell_i, cell_j, distance, group
    """
    ...

def knn_graph(
    coords: Coords,
    barcodes: list[str],
    k: int,
    group: str = "",
) -> Table:
    """
    Build a k-nearest-neighbour graph.

    Returns
    -------
    list[dict]  —  keys: cell_i, cell_j, distance, group
    """
    ...

def graph_stats(
    coords: Coords,
    barcodes: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Compute per-cell neighbour count (degree) for a radius graph.

    Returns
    -------
    list[dict]  —  keys: cell_i, n_neighbors, group
    """
    ...

# ─── cell-type interactions ───────────────────────────────────────────────────

def count_interactions(
    coords: Coords,
    barcodes: list[str],
    cell_types: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Count cell-type pair interactions within a radius (undirected, canonical ordering).

    Returns
    -------
    list[dict]  —  keys: group, cell_type_a, cell_type_b, count
    """
    ...

def interaction_stats(
    coords: Coords,
    barcodes: list[str],
    cell_types: list[str],
    radius: float,
    n_perms: int = 1000,
    seed: int = 42,
    group: str = "",
) -> Table:
    """
    Permutation-based interaction enrichment test.

    Returns
    -------
    list[dict]  —  keys: group, cell_type_a, cell_type_b, observed,
                   expected_mean, expected_std, z_score, p_value
    """
    ...

# ─── neighbourhood composition ────────────────────────────────────────────────

def neighborhood_composition(
    coords: Coords,
    barcodes: list[str],
    cell_types: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Per-cell neighbourhood composition (fraction of each cell type within radius).

    Returns
    -------
    list[dict]  —  keys: cell_i, cell_type, fraction, group
    """
    ...

# ─── spatial autocorrelation ──────────────────────────────────────────────────

def morans(
    coords: Coords,
    values: Matrix,
    feature_names: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Global Moran's I for each feature column (analytical significance).

    Returns
    -------
    list[dict]  —  keys: feature, moran_i, expected_i, variance_i, z_score,
                   group
    """
    ...

def lisa(
    coords: Coords,
    barcodes: list[str],
    values: Matrix,
    feature_names: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Local Moran's I (LISA) per cell × feature with BH FDR correction.

    Returns
    -------
    list[dict]  —  keys: cell_i, feature, local_i, z_score, p_value,
                   q_value_bh, group
    """
    ...

def geary(
    coords: Coords,
    values: Matrix,
    feature_names: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Geary's C for each feature column.  C<1 → positive autocorrelation.

    Returns
    -------
    list[dict]  —  keys: feature, geary_c, expected_c, variance_c, z_score,
                   group
    """
    ...

def bivariate_morans(
    coords: Coords,
    values: Matrix,
    feature_names: list[str],
    radius: float,
    group: str = "",
) -> Table:
    """
    Bivariate Moran's I for all pairs of feature columns.

    Returns
    -------
    list[dict]  —  keys: feature_a, feature_b, bivariate_i, z_score, group
    """
    ...

# ─── dimensionality reduction / clustering ────────────────────────────────────

def nmf_factorize(
    expression: Matrix,
    barcodes: list[str],
    gene_names: list[str],
    n_components: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 42,
) -> dict[str, object]:
    """
    NMF factorisation (Lee & Seung multiplicative updates).

    Accepts dense numpy arrays or scipy sparse matrices.

    Returns
    -------
    dict with keys:
      W  : np.ndarray float32 (N × K)
      H  : np.ndarray float32 (K × G)
      n_iter, final_error, component_variances, barcodes, gene_names
    """
    ...

def gmm_cluster(
    features: Matrix,
    barcodes: list[str],
    n_components: int = 10,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict[str, object]:
    """
    GMM spatial niche detection (EM, k-means++ init, diagonal covariance).

    Returns
    -------
    dict with keys:
      labels : list[int] (0-indexed hard assignment)
      barcodes, log_likelihood, bic, aic, n_iter
    """
    ...

# ─── spatial aggregation ──────────────────────────────────────────────────────

def multiscale_aggregate(
    coords: Coords,
    barcodes: list[str],
    embedding: Matrix,
    radii: list[float],
    include_self: bool = True,
    weighting: str = "gaussian",
    weighting_param: Optional[float] = None,
    group: str = "",
) -> npt.NDArray[np.float64]:
    """
    Aggregate an embedding at multiple spatial scales and concatenate (CellCharter-style).

    For each cell, stacks: [0-hop embedding | agg at radii[0] | agg at radii[1] | ...]
    The result captures local tissue context, not just cell identity.
    Feed directly into ``gmm_cluster`` for compartment / niche detection.

    Parameters
    ----------
    radii : e.g. ``[50.0, 150.0, 300.0]``  — neighbourhood radii in µm
    include_self : prepend the raw 0-hop embedding as the first block
    weighting : ``"uniform"``, ``"gaussian"``, ``"exponential"``, ``"inverse_distance"``
    weighting_param : sigma / decay / epsilon; defaults to min(radii) / 2

    Returns
    -------
    np.ndarray float64, shape (N, D × n_blocks)
        n_blocks = len(radii) + (1 if include_self else 0)
    """
    ...

def aggregate(
    coords: Coords,
    barcodes: list[str],
    embedding: Matrix,
    graph_mode: str,
    graph_param: float,
    weighting: str = "uniform",
    weighting_param: Optional[float] = None,
    group: str = "",
) -> Table:
    """
    Distance-weighted spatial aggregation of an embedding.

    Parameters
    ----------
    graph_mode : ``"radius"`` or ``"knn"``
    graph_param : radius in µm, or k for knn
    weighting : ``"uniform"``, ``"gaussian"``, ``"exponential"``,
                or ``"inverse_distance"``
    weighting_param : sigma / decay / epsilon depending on weighting

    Returns
    -------
    list[dict]  —  keys: cell_i, dim, value, group
        Pivot to wide: ``df.pivot(index="cell_i", columns="dim", values="value")``
    """
    ...

# ─── niche analysis ───────────────────────────────────────────────────────────

def niche_markers(
    expression: Matrix,
    gene_names: list[str],
    niche_labels: list[int] | npt.NDArray[np.int64],
) -> Table:
    """
    One-vs-rest Wilcoxon marker genes for each spatial niche.

    Returns
    -------
    list[dict]  —  keys: niche, gene, mean_niche, mean_rest, log2fc,
                   z_score, p_value, q_value_bh
    """
    ...

def niche_transitions(
    coords: Coords,
    niche_labels: list[int] | npt.NDArray[np.int64],
    radius: float,
    group: str = "",
) -> Table:
    """
    Niche co-occurrence counts within a radius.

    Returns
    -------
    list[dict]  —  keys: niche_a, niche_b, count, fraction, group
    """
    ...

def niche_transition_stats(
    coords: Coords,
    niche_labels: list[int] | npt.NDArray[np.int64],
    radius: float,
    n_perms: int = 1000,
    seed: int = 42,
    group: str = "",
) -> Table:
    """
    Permutation-based niche co-occurrence enrichment.

    Returns
    -------
    list[dict]  —  keys: niche_a, niche_b, observed, expected_mean,
                   expected_std, z_score, p_value, group
    """
    ...

# ─── spatial patterns ─────────────────────────────────────────────────────────

def neighborhood_rings(
    coords: Coords,
    barcodes: list[str],
    cell_types: list[str],
    ring_edges: list[float],
    include_zeros: bool = False,
    group: str = "",
) -> Table:
    """
    Cell-type composition in concentric distance rings.

    Parameters
    ----------
    ring_edges : e.g. ``[0, 20, 50, 100, 200]`` → 4 rings

    Returns
    -------
    list[dict]  —  keys: cell_i, ring_inner, ring_outer, cell_type, count,
                   fraction, group
    """
    ...

def local_correlation(
    coords: Coords,
    barcodes: list[str],
    values_a: Vector | list[float],
    values_b: Vector | list[float],
    feature_a: str,
    feature_b: str,
    radius: float,
    group: str = "",
) -> Table:
    """
    Per-cell Pearson correlation between two features within a radius.

    Returns
    -------
    list[dict]  —  keys: cell_i, feature_a, feature_b, local_r, n_neighbors,
                   group
    """
    ...

def ripley_l(
    coords: Coords,
    cell_types: list[str],
    target_type: str,
    radii: list[float],
    n_sims: int = 199,
    seed: int = 42,
    group: str = "",
) -> Table:
    """
    Ripley's L(r) with Monte Carlo CSR confidence envelope.

    Returns
    -------
    list[dict]  —  keys: cell_type, r, l_r, l_lo, l_hi, group
    """
    ...

def cross_ripley_l(
    coords: Coords,
    cell_types: list[str],
    type_a: str,
    type_b: str,
    radii: list[float],
    n_sims: int = 199,
    seed: int = 42,
    group: str = "",
) -> Table:
    """
    Cross-Ripley L(r) with Monte Carlo CSR envelope.

    Returns
    -------
    list[dict]  —  keys: type_a, type_b, r, l_cross, l_lo, l_hi, group
    """
    ...

# ─── preprocessing ────────────────────────────────────────────────────────────

def normalize_total(
    expression: Matrix,
    target_sum: float = 1e4,
) -> npt.NDArray[np.float32]:
    """Normalize each cell to ``target_sum`` total counts."""
    ...

def log1p(
    expression: Matrix,
) -> npt.NDArray[np.float32]:
    """Apply log(1 + x) element-wise."""
    ...

def filter_cells(
    expression: Matrix,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_counts: Optional[float] = None,
    max_counts: Optional[float] = None,
) -> list[bool]:
    """Return a boolean mask for cells passing QC thresholds."""
    ...

def filter_genes(
    expression: Matrix,
    min_cells: int = 10,
) -> list[bool]:
    """Return a boolean mask for genes expressed in >= ``min_cells`` cells."""
    ...

def scale(
    expression: Matrix,
    max_value: Optional[float] = None,
) -> npt.NDArray[np.float32]:
    """Z-score each gene (column) to mean=0, std=1."""
    ...

def highly_variable_genes(
    expression: Matrix,
    gene_names: list[str],
    n_top_genes: int = 2000,
    n_bins: int = 20,
) -> Table:
    """
    Identify highly variable genes (Seurat v1 / scanpy default).

    Returns
    -------
    list[dict]  —  keys: gene, mean, variance, dispersion, dispersion_norm,
                   highly_variable
    """
    ...
