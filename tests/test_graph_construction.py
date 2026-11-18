from __future__ import annotations

import math

import torch

from chromatin_gnn.data import LOCUS_COUNT, NODE_FEATURE_DIM, build_graph


def synthetic_trace() -> tuple[list[bool], list[tuple[float, float, float] | None]]:
    mask = [False] * LOCUS_COUNT
    coordinates: list[tuple[float, float, float] | None] = [None] * LOCUS_COUNT
    for index, xyz in {
        0: (0.0, 0.0, 0.0),
        1: (10.0, 0.0, 0.0),
        2: (100.0, 0.0, 0.0),
        3: (500.0, 0.0, 0.0),
    }.items():
        mask[index] = True
        coordinates[index] = xyz
    return mask, coordinates


def test_graph_uses_76_node_features_and_excludes_adjacent_loci() -> None:
    mask, coordinates = synthetic_trace()
    graph = build_graph(mask, coordinates, label=0, run_group=1, graph_id=0)

    assert graph.x.shape == (4, NODE_FEATURE_DIM)
    assert NODE_FEATURE_DIM == 76
    edge_pairs = set(map(tuple, graph.edge_index.t().tolist()))
    assert edge_pairs == {(0, 2), (2, 0)}
    assert (0, 1) not in edge_pairs
    assert (1, 2) not in edge_pairs


def test_degree_is_normalized_by_eligible_nonadjacent_partners() -> None:
    mask, coordinates = synthetic_trace()
    graph = build_graph(mask, coordinates, label=0, run_group=1, graph_id=0)
    normalized_degree_column = LOCUS_COUNT + 1

    assert torch.isclose(
        graph.x[0, normalized_degree_column], torch.tensor(0.5)
    )
    assert torch.isclose(
        graph.x[2, normalized_degree_column], torch.tensor(1.0)
    )


def test_radial_features_are_finite() -> None:
    mask, coordinates = synthetic_trace()
    graph = build_graph(mask, coordinates, label=0, run_group=1, graph_id=0)
    radial = graph.x[:, LOCUS_COUNT + 2]
    assert all(math.isfinite(value) for value in radial.tolist())


def test_contact_at_exact_threshold_is_excluded() -> None:
    mask = [False] * LOCUS_COUNT
    coordinates: list[tuple[float, float, float] | None] = [None] * LOCUS_COUNT
    mask[0] = mask[2] = True
    coordinates[0] = (0.0, 0.0, 0.0)
    coordinates[2] = (238.0, 0.0, 0.0)
    graph = build_graph(mask, coordinates, label=0, run_group=1, graph_id=0)
    assert graph.edge_index.shape == (2, 0)
