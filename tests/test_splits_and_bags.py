from __future__ import annotations

import torch
from torch_geometric.data import Data

from chromatin_gnn.data import make_balanced_bags, make_split


def synthetic_dataset() -> list[Data]:
    dataset = []
    graph_id = 0
    for label in (0, 1):
        for run_group in (1, 2, 3):
            for _ in range(40):
                dataset.append(
                    Data(
                        x=torch.zeros((1, 76)),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        y=torch.tensor([float(label)]),
                        run_group=torch.tensor([run_group]),
                        graph_id=torch.tensor([graph_id]),
                    )
                )
                graph_id += 1
    return dataset


def test_split_is_disjoint() -> None:
    dataset = synthetic_dataset()
    train, validation, test = make_split(dataset, seed=224)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(train) | set(validation) | set(test) == set(range(len(dataset)))


def test_every_bag_has_one_class_and_run_group() -> None:
    dataset = synthetic_dataset()
    bags = make_balanced_bags(dataset, list(range(len(dataset))), bag_size=10, seed=7)
    assert bags
    labels = []
    for bag in bags:
        assert len(bag) == 10
        bag_labels = {int(dataset[index].y.item()) for index in bag}
        bag_groups = {int(dataset[index].run_group.item()) for index in bag}
        assert len(bag_labels) == 1
        assert len(bag_groups) == 1
        labels.append(next(iter(bag_labels)))
    assert labels.count(0) == labels.count(1)
