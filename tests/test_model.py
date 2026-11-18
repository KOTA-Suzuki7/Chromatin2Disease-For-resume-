from __future__ import annotations

import torch
from torch_geometric.data import Batch

from chromatin_gnn.data import LOCUS_COUNT, build_graph
from chromatin_gnn.model import InterpretableBagGNN, graph_to_bag_logits


def small_graph(graph_id: int):
    mask = [False] * LOCUS_COUNT
    coordinates: list[tuple[float, float, float] | None] = [None] * LOCUS_COUNT
    for index, xyz in {
        0: (0.0, 0.0, 0.0),
        2: (100.0, 0.0, 0.0),
        4: (400.0, 0.0, 0.0),
    }.items():
        mask[index] = True
        coordinates[index] = xyz
    return build_graph(mask, coordinates, graph_id % 2, 1, graph_id)


def test_signed_contributions_exactly_reconstruct_graph_logits() -> None:
    batch = Batch.from_data_list([small_graph(0), small_graph(1)])
    model = InterpretableBagGNN(hidden_dim=8)
    model.eval()
    logits, attention, _, contribution = model(
        batch.x, batch.edge_index, batch.batch, return_explanations=True
    )

    rebuilt = torch.stack(
        [contribution[batch.batch == index].sum() for index in range(2)]
    ) + model.graph_bias
    assert torch.allclose(logits, rebuilt)
    assert torch.allclose(
        torch.stack([attention[batch.batch == index].sum() for index in range(2)]),
        torch.ones(2),
    )


def test_bag_logit_is_mean_of_graph_logits() -> None:
    graph_logits = torch.tensor([1.0, 3.0, -2.0, 2.0])
    assert torch.equal(graph_to_bag_logits(graph_logits, 2), torch.tensor([2.0, 0.0]))


def test_incomplete_bag_is_rejected() -> None:
    try:
        graph_to_bag_logits(torch.tensor([1.0, 2.0, 3.0]), 2)
    except ValueError:
        return
    raise AssertionError("Expected an incomplete bag to raise ValueError")
