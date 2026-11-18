"""Load Hafner et al. FOF-CT tables and construct contact graphs."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "hafner2023"

CONTACT_THRESHOLD_NM = 238.0
MIN_NODES = 37
LOCI = tuple(range(50_590_004, 52_750_076 + 30_001, 30_001))
LOCUS_INDEX = {start: index for index, start in enumerate(LOCI)}
LOCUS_COUNT = len(LOCI)
PAPER_CBS_READOUTS = frozenset({15, 23, 37, 46, 57})
NODE_FEATURE_DIM = LOCUS_COUNT + 3
NODE_FEATURE_NAMES = (
    "locus_identity_one_hot_73",
    "paper_cbs_indicator",
    "normalized_contact_degree",
    "normalized_radial_position",
)

# label, run group, accession-based filename
SOURCE_FILES = (
    (0, 1, "4DNFIZWUQTXF.csv"),
    (0, 2, "4DNFIKVZTN9K.csv"),
    (0, 3, "4DNFID7HZ8LX.csv"),
    (1, 1, "4DNFIJYFR8FT.csv"),
    (1, 2, "4DNFIIOVHG1I.csv"),
    (1, 3, "4DNFI4VLY3WU.csv"),
)

Coordinates = list[tuple[float, float, float] | None]
Trace = tuple[list[bool], Coordinates]


def _read_fof_rows(path: Path) -> list[dict[str, str]]:
    """Read a FOF-CT table whose column declaration is embedded in comments."""

    columns: list[str] | None = None
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for values in reader:
            if not values:
                continue
            first = values[0]
            if first.startswith("##columns=("):
                declaration = first.removeprefix("##columns=(")
                declaration += "," + ",".join(values[1:])
                columns = [value.strip() for value in declaration.rstrip(")").split(",")]
                continue
            if first.startswith("#"):
                continue
            if columns is None:
                raise ValueError(f"No ##columns declaration before data in {path}")
            if len(values) != len(columns):
                raise ValueError(f"Malformed row in {path}: expected {len(columns)} values")
            rows.append(dict(zip(columns, values)))
    if columns is None:
        raise ValueError(f"No ##columns declaration in {path}")
    required = {"Trace_ID", "X", "Y", "Z", "Chrom", "Chrom_Start"}
    if not required.issubset(columns):
        raise ValueError(f"Missing required columns in {path}: {required - set(columns)}")
    return rows


def load_traces(path: Path) -> list[Trace]:
    """Return canonical observation masks and coordinates for every trace."""

    coordinates: dict[int, Coordinates] = {}
    masks: dict[int, list[bool]] = {}
    for row in _read_fof_rows(path):
        if row["Chrom"] != "chr6":
            continue
        locus_index = LOCUS_INDEX.get(int(row["Chrom_Start"]))
        if locus_index is None:
            continue
        xyz = tuple(float(row[axis]) for axis in ("X", "Y", "Z"))
        if not all(math.isfinite(value) for value in xyz):
            continue
        trace_id = int(row["Trace_ID"])
        if trace_id not in masks:
            masks[trace_id] = [False] * LOCUS_COUNT
            coordinates[trace_id] = [None] * LOCUS_COUNT
        masks[trace_id][locus_index] = True
        coordinates[trace_id][locus_index] = xyz
    return [(masks[trace_id], coordinates[trace_id]) for trace_id in sorted(masks)]


def build_graph(
    mask: list[bool],
    coordinates: Coordinates,
    label: int,
    run_group: int,
    graph_id: int,
) -> Data:
    """Construct one variable-size graph from a chromosome trace."""

    canonical_indices = [index for index, observed in enumerate(mask) if observed]
    if not canonical_indices:
        raise ValueError("Cannot construct a graph without observed loci")
    local_index = {canonical: local for local, canonical in enumerate(canonical_indices)}
    xyz = np.asarray([coordinates[index] for index in canonical_indices], dtype=np.float32)
    centroid = xyz.mean(axis=0)
    radii = np.linalg.norm(xyz - centroid, axis=1)
    radius_of_gyration = float(np.sqrt(np.mean(np.square(radii))))
    if radius_of_gyration <= 0:
        radius_of_gyration = 1.0

    degrees = [0] * len(canonical_indices)
    directed_edges: list[tuple[int, int]] = []
    threshold_squared = CONTACT_THRESHOLD_NM**2
    for offset, first in enumerate(canonical_indices):
        xyz_first = coordinates[first]
        assert xyz_first is not None
        for second in canonical_indices[offset + 1 :]:
            if second - first == 1:
                continue
            xyz_second = coordinates[second]
            assert xyz_second is not None
            distance_squared = sum((a - b) ** 2 for a, b in zip(xyz_first, xyz_second))
            if distance_squared >= threshold_squared:
                continue
            first_local = local_index[first]
            second_local = local_index[second]
            directed_edges.extend(((first_local, second_local), (second_local, first_local)))
            degrees[first_local] += 1
            degrees[second_local] += 1

    edge_index = (
        torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        if directed_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    node_features: list[list[float]] = []
    for local, canonical in enumerate(canonical_indices):
        one_hot = [0.0] * LOCUS_COUNT
        one_hot[canonical] = 1.0
        readout = canonical + 4
        cbs_indicator = float(readout in PAPER_CBS_READOUTS)
        adjacent_observed = int(canonical - 1 in local_index) + int(
            canonical + 1 in local_index
        )
        eligible_partners = max(1, len(canonical_indices) - 1 - adjacent_observed)
        normalized_degree = degrees[local] / eligible_partners
        normalized_radius = float(radii[local] / radius_of_gyration)
        node_features.append(
            one_hot + [cbs_indicator, normalized_degree, normalized_radius]
        )

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([float(label)], dtype=torch.float32),
        locus_id=torch.tensor(canonical_indices, dtype=torch.long),
        run_group=torch.tensor([run_group], dtype=torch.long),
        graph_id=torch.tensor([graph_id], dtype=torch.long),
        observed_mask=torch.tensor([mask], dtype=torch.bool),
    )


def load_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    minimum_nodes: int = MIN_NODES,
) -> tuple[list[Data], list[dict[str, object]]]:
    """Load all six files and retain traces meeting the coverage threshold."""

    dataset: list[Data] = []
    provenance: list[dict[str, object]] = []
    graph_id = 0
    for label, run_group, filename in SOURCE_FILES:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python scripts/download_data.py` first."
            )
        traces = load_traces(path)
        retained = 0
        for mask, coordinates in traces:
            if sum(mask) < minimum_nodes:
                continue
            dataset.append(build_graph(mask, coordinates, label, run_group, graph_id))
            graph_id += 1
            retained += 1
        provenance.append(
            {
                "file": filename,
                "condition": "ctcf_depleted" if label else "untreated",
                "run_group": run_group,
                "raw_graphs": len(traces),
                "retained_graphs": retained,
            }
        )
    return dataset, provenance


def make_split(
    dataset: list[Data], seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Create a 70/15/15 trace split stratified by class and run group."""

    indices = np.arange(len(dataset))
    labels = np.asarray([int(data.y.item()) for data in dataset])
    groups = np.asarray([int(data.run_group.item()) for data in dataset])
    strata = np.asarray([f"{label}_{group}" for label, group in zip(labels, groups)])
    train, remainder = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=strata
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=strata[remainder],
    )
    return train.tolist(), validation.tolist(), test.tolist()


def make_balanced_bags(
    dataset: list[Data],
    indices: list[int],
    bag_size: int,
    seed: int,
) -> list[list[int]]:
    """Create disjoint, class-balanced bags within each experimental run group."""

    rng = random.Random(seed)
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in indices:
        label = int(dataset[index].y.item())
        run_group = int(dataset[index].run_group.item())
        grouped[(label, run_group)].append(index)

    bags: list[list[int]] = []
    for run_group in sorted({key[1] for key in grouped}):
        by_label: dict[int, list[list[int]]] = {}
        for label in (0, 1):
            values = grouped[(label, run_group)].copy()
            rng.shuffle(values)
            by_label[label] = [
                values[start : start + bag_size]
                for start in range(0, len(values) - bag_size + 1, bag_size)
            ]
        paired_count = min(len(by_label[0]), len(by_label[1]))
        for label in (0, 1):
            rng.shuffle(by_label[label])
            bags.extend(by_label[label][:paired_count])
    rng.shuffle(bags)
    return bags


def bag_summary(dataset: list[Data], bags: list[list[int]]) -> dict[str, object]:
    labels = [int(dataset[bag[0]].y.item()) for bag in bags]
    groups = [int(dataset[bag[0]].run_group.item()) for bag in bags]
    return {
        "bags": len(bags),
        "graphs": sum(len(bag) for bag in bags),
        "untreated_bags": labels.count(0),
        "ctcf_depleted_bags": labels.count(1),
        "bags_by_run_group": {
            str(group): groups.count(group) for group in sorted(set(groups))
        },
    }
