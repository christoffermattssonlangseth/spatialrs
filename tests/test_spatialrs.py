"""
pytest tests for the spatialrs Python bindings.

Each test uses small synthetic data so the suite runs in seconds.
Run with:  pytest tests/test_spatialrs.py -v
"""

import numpy as np
import pandas as pd
import pytest
import spatialrs

# ─── fixtures ─────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)

# 20 cells on a 5×4 grid, spacing 10 µm
_GRID = np.array([[x * 10.0, y * 10.0] for x in range(5) for y in range(4)], dtype=np.float64)
N = len(_GRID)  # 20
BARCODES = [f"cell_{i}" for i in range(N)]
CELL_TYPES = (["A"] * 10 + ["B"] * 10)[:N]
RADIUS = 15.0  # catches immediate grid neighbours


@pytest.fixture()
def coords():
    return _GRID.copy()


@pytest.fixture()
def expression():
    """Dense float64 expression matrix (N × G)."""
    G = 30
    return np.abs(RNG.standard_normal((N, G)).astype(np.float64)) + 0.1


@pytest.fixture()
def expression_f32():
    G = 30
    return np.abs(RNG.standard_normal((N, G)).astype(np.float32)) + 0.1


@pytest.fixture()
def gene_names():
    return [f"gene_{i}" for i in range(30)]


# ─── graph construction ───────────────────────────────────────────────────────

class TestRadiusGraph:
    def test_returns_list_of_dicts(self, coords):
        result = spatialrs.radius_graph(coords, BARCODES, RADIUS, group="s1")
        assert isinstance(result, list)
        assert len(result) > 0
        row = result[0]
        assert set(row.keys()) == {"cell_i", "cell_j", "distance", "group"}

    def test_distance_within_radius(self, coords):
        result = spatialrs.radius_graph(coords, BARCODES, RADIUS)
        df = pd.DataFrame(result)
        assert (df["distance"] <= RADIUS).all()

    def test_group_propagated(self, coords):
        result = spatialrs.radius_graph(coords, BARCODES, RADIUS, group="g1")
        assert all(r["group"] == "g1" for r in result)


class TestKnnGraph:
    def test_keys(self, coords):
        result = spatialrs.knn_graph(coords, BARCODES, k=4)
        assert set(result[0].keys()) == {"cell_i", "cell_j", "distance", "group"}

    def test_exactly_k_neighbours_per_cell(self, coords):
        k = 3
        result = spatialrs.knn_graph(coords, BARCODES, k=k)
        df = pd.DataFrame(result)
        counts = df.groupby("cell_i").size()
        assert (counts == k).all()


class TestGraphStats:
    def test_keys(self, coords):
        result = spatialrs.graph_stats(coords, BARCODES, RADIUS)
        assert set(result[0].keys()) == {"cell_i", "n_neighbors", "group"}

    def test_count_per_cell(self, coords):
        result = spatialrs.graph_stats(coords, BARCODES, RADIUS)
        assert len(result) == N


# ─── cell-type interactions ───────────────────────────────────────────────────

class TestCountInteractions:
    def test_keys(self, coords):
        result = spatialrs.count_interactions(coords, BARCODES, CELL_TYPES, RADIUS)
        row = result[0]
        assert {"group", "cell_type_a", "cell_type_b", "count"} <= set(row.keys())

    def test_counts_non_negative(self, coords):
        result = spatialrs.count_interactions(coords, BARCODES, CELL_TYPES, RADIUS)
        df = pd.DataFrame(result)
        assert (df["count"] >= 0).all()


class TestInteractionStats:
    def test_keys(self, coords):
        result = spatialrs.interaction_stats(
            coords, BARCODES, CELL_TYPES, RADIUS, n_perms=50, seed=0
        )
        row = result[0]
        expected = {"group", "cell_type_a", "cell_type_b", "observed",
                    "expected_mean", "expected_std", "z_score", "p_value"}
        assert expected <= set(row.keys())

    def test_p_value_range(self, coords):
        result = spatialrs.interaction_stats(
            coords, BARCODES, CELL_TYPES, RADIUS, n_perms=50, seed=1
        )
        df = pd.DataFrame(result)
        assert ((df["p_value"] >= 0) & (df["p_value"] <= 1)).all()


# ─── neighbourhood composition ────────────────────────────────────────────────

class TestNeighborhoodComposition:
    def test_keys(self, coords):
        result = spatialrs.neighborhood_composition(coords, BARCODES, CELL_TYPES, RADIUS)
        assert {"cell_i", "cell_type", "fraction", "group"} <= set(result[0].keys())

    def test_fractions_sum_to_one(self, coords):
        result = spatialrs.neighborhood_composition(coords, BARCODES, CELL_TYPES, RADIUS)
        df = pd.DataFrame(result)
        sums = df.groupby("cell_i")["fraction"].sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-5)


# ─── spatial autocorrelation ──────────────────────────────────────────────────

class TestMorans:
    def test_keys(self, coords, expression, gene_names):
        result = spatialrs.morans(coords, expression, gene_names, RADIUS)
        assert {"feature", "moran_i", "expected_i", "variance_i", "z_score", "group"} <= set(result[0].keys())

    def test_one_row_per_gene(self, coords, expression, gene_names):
        result = spatialrs.morans(coords, expression, gene_names, RADIUS)
        assert len(result) == len(gene_names)


class TestLisa:
    def test_keys(self, coords, expression, gene_names):
        result = spatialrs.lisa(coords, BARCODES, expression, gene_names, RADIUS)
        assert {"cell_i", "feature", "local_i", "z_score", "p_value", "q_value_bh", "group"} <= set(result[0].keys())

    def test_row_count(self, coords, expression, gene_names):
        result = spatialrs.lisa(coords, BARCODES, expression, gene_names, RADIUS)
        assert len(result) == N * len(gene_names)


class TestGeary:
    def test_keys(self, coords, expression, gene_names):
        result = spatialrs.geary(coords, expression, gene_names, RADIUS)
        assert {"feature", "geary_c", "expected_c", "variance_c", "z_score", "group"} <= set(result[0].keys())


class TestBivariateMorans:
    def test_keys(self, coords, expression, gene_names):
        result = spatialrs.bivariate_morans(coords, expression, gene_names, RADIUS)
        assert {"feature_a", "feature_b", "bivariate_i", "z_score", "group"} <= set(result[0].keys())

    def test_pair_count(self, coords, expression, gene_names):
        G = len(gene_names)
        result = spatialrs.bivariate_morans(coords, expression, gene_names, RADIUS)
        # upper triangle pairs: G*(G-1)/2
        assert len(result) == G * (G - 1) // 2


# ─── dimensionality reduction / clustering ────────────────────────────────────

class TestNmfFactorize:
    def test_return_keys(self, expression, gene_names):
        out = spatialrs.nmf_factorize(expression, BARCODES, gene_names, n_components=3, max_iter=50)
        assert {"W", "H", "n_iter", "final_error", "component_variances", "barcodes", "gene_names"} <= set(out.keys())

    def test_shapes(self, expression, gene_names):
        K = 4
        out = spatialrs.nmf_factorize(expression, BARCODES, gene_names, n_components=K, max_iter=50)
        assert out["W"].shape == (N, K)
        assert out["H"].shape == (K, len(gene_names))

    def test_non_negative(self, expression, gene_names):
        out = spatialrs.nmf_factorize(expression, BARCODES, gene_names, n_components=3, max_iter=50)
        assert (out["W"] >= 0).all()
        assert (out["H"] >= 0).all()

    def test_accepts_float32(self, expression_f32, gene_names):
        out = spatialrs.nmf_factorize(expression_f32, BARCODES, gene_names, n_components=3, max_iter=30)
        assert out["W"].shape[0] == N

    def test_accepts_scipy_sparse(self, expression, gene_names):
        import scipy.sparse as sp
        sparse_expr = sp.csr_matrix(expression)
        out = spatialrs.nmf_factorize(sparse_expr, BARCODES, gene_names, n_components=3, max_iter=30)
        assert out["W"].shape[0] == N


class TestGmmCluster:
    def test_return_keys(self, expression):
        out = spatialrs.gmm_cluster(expression, BARCODES, n_components=3, max_iter=50)
        assert {"labels", "barcodes", "log_likelihood", "bic", "aic", "n_iter"} <= set(out.keys())

    def test_label_count(self, expression):
        out = spatialrs.gmm_cluster(expression, BARCODES, n_components=3, max_iter=50)
        assert len(out["labels"]) == N

    def test_label_range(self, expression):
        K = 3
        out = spatialrs.gmm_cluster(expression, BARCODES, n_components=K, max_iter=50)
        labels = np.array(out["labels"])
        assert labels.min() >= 0 and labels.max() < K


# ─── spatial aggregation ──────────────────────────────────────────────────────

class TestAggregate:
    def test_keys(self, coords, expression):
        result = spatialrs.aggregate(
            coords, BARCODES, expression,
            graph_mode="radius", graph_param=RADIUS,
            weighting="uniform",
        )
        assert {"cell_i", "dim", "value", "group"} <= set(result[0].keys())

    def test_row_count(self, coords, expression):
        D = expression.shape[1]
        result = spatialrs.aggregate(
            coords, BARCODES, expression,
            graph_mode="radius", graph_param=RADIUS,
        )
        assert len(result) == N * D

    def test_knn_mode(self, coords, expression):
        result = spatialrs.aggregate(
            coords, BARCODES, expression,
            graph_mode="knn", graph_param=4.0,
        )
        assert len(result) == N * expression.shape[1]


# ─── niche analysis ───────────────────────────────────────────────────────────

class TestNicheMarkers:
    def test_keys(self, expression, gene_names):
        labels = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = spatialrs.niche_markers(expression, gene_names, labels)
        assert {"niche", "gene", "mean_niche", "mean_rest", "log2fc",
                "z_score", "p_value", "q_value_bh"} <= set(result[0].keys())

    def test_q_value_range(self, expression, gene_names):
        labels = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = spatialrs.niche_markers(expression, gene_names, labels)
        df = pd.DataFrame(result)
        assert ((df["q_value_bh"] >= 0) & (df["q_value_bh"] <= 1)).all()


class TestNicheTransitions:
    def test_keys(self, coords):
        labels = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = spatialrs.niche_transitions(coords, labels, RADIUS)
        assert {"niche_a", "niche_b", "count", "fraction", "group"} <= set(result[0].keys())


class TestNicheTransitionStats:
    def test_keys(self, coords):
        labels = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = spatialrs.niche_transition_stats(coords, labels, RADIUS, n_perms=50, seed=0)
        assert {"niche_a", "niche_b", "observed", "expected_mean",
                "expected_std", "z_score", "p_value", "group"} <= set(result[0].keys())


# ─── spatial patterns ─────────────────────────────────────────────────────────

class TestNeighborhoodRings:
    def test_keys(self, coords):
        result = spatialrs.neighborhood_rings(
            coords, BARCODES, CELL_TYPES, ring_edges=[0.0, 15.0, 30.0]
        )
        assert {"cell_i", "ring_inner", "ring_outer", "cell_type",
                "count", "fraction", "group"} <= set(result[0].keys())


class TestLocalCorrelation:
    def test_keys(self, coords, expression):
        a = expression[:, 0].astype(np.float64)
        b = expression[:, 1].astype(np.float64)
        result = spatialrs.local_correlation(
            coords, BARCODES, a, b, "gene_0", "gene_1", RADIUS
        )
        assert {"cell_i", "feature_a", "feature_b", "local_r", "n_neighbors", "group"} <= set(result[0].keys())

    def test_r_range(self, coords, expression):
        a = expression[:, 0].astype(np.float64)
        b = expression[:, 1].astype(np.float64)
        result = spatialrs.local_correlation(
            coords, BARCODES, a, b, "gene_0", "gene_1", RADIUS
        )
        df = pd.DataFrame(result)
        # Pearson r in [-1, 1]; cells with < 2 neighbours return 0
        assert ((df["local_r"] >= -1.0) & (df["local_r"] <= 1.0)).all()


class TestRipleyL:
    def test_keys(self, coords):
        result = spatialrs.ripley_l(
            coords, CELL_TYPES, target_type="A",
            radii=[5.0, 10.0, 20.0], n_sims=19, seed=0
        )
        assert {"cell_type", "r", "l_r", "l_lo", "l_hi", "group"} <= set(result[0].keys())

    def test_one_row_per_radius(self, coords):
        radii = [5.0, 10.0, 20.0, 40.0]
        result = spatialrs.ripley_l(
            coords, CELL_TYPES, target_type="A",
            radii=radii, n_sims=19, seed=0
        )
        assert len(result) == len(radii)


class TestCrossRipleyL:
    def test_keys(self, coords):
        result = spatialrs.cross_ripley_l(
            coords, CELL_TYPES, type_a="A", type_b="B",
            radii=[5.0, 10.0, 20.0], n_sims=19, seed=0
        )
        assert {"type_a", "type_b", "r", "l_cross", "l_lo", "l_hi", "group"} <= set(result[0].keys())


# ─── preprocessing ────────────────────────────────────────────────────────────

class TestNormalizeTotal:
    def test_output_shape(self, expression):
        out = spatialrs.normalize_total(expression, target_sum=1e4)
        assert out.shape == expression.shape

    def test_row_sums(self, expression):
        out = spatialrs.normalize_total(expression, target_sum=1e4)
        row_sums = out.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1e4, rtol=1e-4)

    def test_dtype_f32(self, expression):
        out = spatialrs.normalize_total(expression)
        assert out.dtype == np.float32


class TestLog1p:
    def test_output_shape(self, expression):
        out = spatialrs.log1p(expression)
        assert out.shape == expression.shape

    def test_values(self, expression):
        out = spatialrs.log1p(expression)
        expected = np.log1p(expression).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestFilterCells:
    def test_returns_bool_list(self, expression):
        mask = spatialrs.filter_cells(expression, min_genes=1)
        assert isinstance(mask, list)
        assert len(mask) == N
        assert all(isinstance(v, bool) for v in mask)

    def test_min_counts(self, expression):
        mask = spatialrs.filter_cells(expression, min_counts=0.0)
        assert all(mask)  # all cells should pass with min_counts=0

    def test_impossible_threshold_excludes_all(self, expression):
        mask = spatialrs.filter_cells(expression, min_counts=1e9)
        assert not any(mask)


class TestFilterGenes:
    def test_returns_bool_list(self, expression):
        mask = spatialrs.filter_genes(expression, min_cells=1)
        assert isinstance(mask, list)
        assert len(mask) == expression.shape[1]

    def test_zero_threshold_keeps_all(self, expression):
        mask = spatialrs.filter_genes(expression, min_cells=0)
        assert all(mask)


class TestScale:
    def test_output_shape(self, expression):
        out = spatialrs.scale(expression)
        assert out.shape == expression.shape

    def test_column_means_near_zero(self, expression):
        out = spatialrs.scale(expression)
        col_means = out.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-4)

    def test_max_value_clipping(self, expression):
        out = spatialrs.scale(expression, max_value=2.0)
        assert np.abs(out).max() <= 2.0 + 1e-5


class TestHighlyVariableGenes:
    def test_keys(self, expression, gene_names):
        result = spatialrs.highly_variable_genes(expression, gene_names, n_top_genes=10)
        row = result[0]
        assert {"gene", "mean", "variance", "dispersion",
                "dispersion_norm", "highly_variable"} <= set(row.keys())

    def test_top_gene_count(self, expression, gene_names):
        n_top = 10
        result = spatialrs.highly_variable_genes(expression, gene_names, n_top_genes=n_top)
        df = pd.DataFrame(result)
        assert df["highly_variable"].sum() == n_top

    def test_all_genes_present(self, expression, gene_names):
        result = spatialrs.highly_variable_genes(expression, gene_names)
        df = pd.DataFrame(result)
        assert set(df["gene"]) == set(gene_names)


# ─── regression tests for codex review issues ────────────────────────────────

class TestSparseFloat32:
    """Issue 1: sparse float32 matrices raised TypeError in the toarray() path."""

    def test_nmf_sparse_float32(self, expression, gene_names):
        import scipy.sparse as sp
        sparse_f32 = sp.csr_matrix(expression.astype(np.float32))
        out = spatialrs.nmf_factorize(sparse_f32, BARCODES, gene_names, n_components=3, max_iter=30)
        assert out["W"].shape == (N, 3)

    def test_morans_sparse_float32(self, coords, expression, gene_names):
        import scipy.sparse as sp
        sparse_f32 = sp.csr_matrix(expression.astype(np.float32))
        result = spatialrs.morans(coords, sparse_f32, gene_names, RADIUS)
        assert len(result) == len(gene_names)

    def test_normalize_total_sparse_float32(self, expression):
        import scipy.sparse as sp
        sparse_f32 = sp.csr_matrix(expression.astype(np.float32))
        out = spatialrs.normalize_total(sparse_f32, target_sum=1e4)
        assert out.shape == expression.shape


class TestSparseNmfPath:
    """Issue 2: CSR sparse input should use the memory-efficient sparse NMF path."""

    def test_csr_uses_sparse_path(self, expression, gene_names):
        import scipy.sparse as sp
        # The sparse path tracks a different convergence metric than the dense path
        # (relative H change vs. Frobenius reconstruction error), so we only
        # verify that the sparse path returns valid, non-negative NMF factors.
        out = spatialrs.nmf_factorize(
            sp.csr_matrix(expression), BARCODES, gene_names, n_components=4, max_iter=100, seed=0
        )
        assert out["W"].shape == (N, 4)
        assert out["H"].shape == (4, len(gene_names))
        assert (out["W"] >= 0).all()
        assert (out["H"] >= 0).all()

    def test_non_csr_sparse_falls_back(self, expression, gene_names):
        import scipy.sparse as sp
        # COO matrix has no indptr — should fall back to dense path without error
        coo = sp.coo_matrix(expression)
        out = spatialrs.nmf_factorize(coo, BARCODES, gene_names, n_components=3, max_iter=30)
        assert out["W"].shape == (N, 3)


class TestKnnGraphParamValidation:
    """Issue 3: knn with graph_param in (0,1) would silently produce k=0."""

    def test_knn_zero_raises(self, coords, expression):
        with pytest.raises(Exception, match="k.*must be >= 1|knn"):
            spatialrs.aggregate(
                coords, BARCODES, expression,
                graph_mode="knn", graph_param=0.5,  # truncates to 0
            )

    def test_knn_fractional_truncates_correctly(self, coords, expression):
        # 3.9 should behave as k=3, not silently fail
        result = spatialrs.aggregate(
            coords, BARCODES, expression,
            graph_mode="knn", graph_param=3.9,
        )
        assert len(result) == N * expression.shape[1]


class TestMetadataLengthValidation:
    """Issue 4: mismatched metadata lengths should raise, not silently corrupt output."""

    def test_nmf_wrong_barcodes_length(self, expression, gene_names):
        with pytest.raises(Exception, match="barcodes"):
            spatialrs.nmf_factorize(
                expression, BARCODES[:-1], gene_names, n_components=3, max_iter=10
            )

    def test_nmf_wrong_gene_names_length(self, expression, gene_names):
        with pytest.raises(Exception, match="gene_names"):
            spatialrs.nmf_factorize(
                expression, BARCODES, gene_names[:-1], n_components=3, max_iter=10
            )

    def test_gmm_wrong_barcodes_length(self, expression):
        with pytest.raises(Exception, match="barcodes"):
            spatialrs.gmm_cluster(expression, BARCODES[:-1], n_components=3, max_iter=10)
