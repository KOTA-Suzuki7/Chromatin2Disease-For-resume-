"""Interpretable bag-level graph neural network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv
from torch_geometric.utils import scatter, softmax

from .data import NODE_FEATURE_DIM


class InterpretableBagGNN(nn.Module):
    """One sum-message layer with an exactly additive attention readout."""

    def __init__(self, hidden_dim: int = 8, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = dropout
        self.message_layer = GraphConv(
            in_channels=NODE_FEATURE_DIM,
            out_channels=hidden_dim,
            aggr="add",
            bias=True,
        )
        self.readout_attention = nn.Linear(hidden_dim, 1)
        self.node_score = nn.Linear(hidden_dim, 1, bias=False)
        self.graph_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_explanations: bool = False,
    ):
        hidden = F.elu(self.message_layer(x, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        attention = softmax(self.readout_attention(hidden).squeeze(-1), batch)
        signed_node_score = self.node_score(hidden).squeeze(-1)
        contribution = attention * signed_node_score
        graph_logits = (
            scatter(contribution, batch, dim=0, reduce="sum") + self.graph_bias
        )
        if return_explanations:
            return graph_logits, attention, signed_node_score, contribution
        return graph_logits


def graph_to_bag_logits(graph_logits: torch.Tensor, bag_size: int) -> torch.Tensor:
    """Average consecutive graph logits within fixed-size bags."""

    if graph_logits.numel() % bag_size:
        raise ValueError("Every minibatch bag must contain the fixed bag size")
    return graph_logits.reshape(-1, bag_size).mean(dim=1)
