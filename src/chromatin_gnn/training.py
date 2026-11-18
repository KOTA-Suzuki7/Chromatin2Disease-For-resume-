"""Training, evaluation, and exact locus-level interpretation utilities."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from .data import (
    CONTACT_THRESHOLD_NM,
    DEFAULT_DATA_DIR,
    LOCI,
    LOCUS_COUNT,
    MIN_NODES,
    NODE_FEATURE_NAMES,
    PAPER_CBS_READOUTS,
    PROJECT_ROOT,
    bag_summary,
    load_dataset,
    make_balanced_bags,
    make_split,
)
from .model import InterpretableBagGNN, graph_to_bag_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iter_bag_minibatches(
    dataset: list[Data],
    bags: list[list[int]],
    bags_per_batch: int,
    shuffle: bool,
    seed: int,
):
    order = list(range(len(bags)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), bags_per_batch):
        selected = [bags[index] for index in order[start : start + bags_per_batch]]
        graphs = [dataset[index] for bag in selected for index in bag]
        graph_batch = Batch.from_data_list(graphs)
        labels = torch.tensor(
            [float(dataset[bag[0]].y.item()) for bag in selected],
            dtype=torch.float32,
        )
        yield graph_batch, labels, selected


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
    }


@torch.no_grad()
def evaluate(
    model: InterpretableBagGNN,
    dataset: list[Data],
    bags: list[list[int]],
    bag_size: int,
    bags_per_batch: int,
    collect_nodes: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]] | None, float]:
    model.eval()
    bag_labels: list[float] = []
    bag_logits_all: list[float] = []
    graph_labels: list[float] = []
    graph_logits_all: list[float] = []
    node_records: dict[str, list[float]] | None = (
        defaultdict(list) if collect_nodes else None
    )
    reconstruction_error = 0.0

    for graph_batch, labels, _ in iter_bag_minibatches(
        dataset, bags, bags_per_batch, False, 0
    ):
        output = model(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch,
            return_explanations=collect_nodes,
        )
        if collect_nodes:
            graph_logits, attention, signed_score, contribution = output
            assert node_records is not None
            rebuilt = (
                scatter(contribution, graph_batch.batch, dim=0, reduce="sum")
                + model.graph_bias
            )
            reconstruction_error = max(
                reconstruction_error,
                float(torch.max(torch.abs(rebuilt - graph_logits)).item()),
            )
            node_labels = graph_batch.y[graph_batch.batch]
            node_records["locus_id"].extend(graph_batch.locus_id.cpu().tolist())
            node_records["label"].extend(node_labels.cpu().tolist())
            node_records["attention"].extend(attention.cpu().tolist())
            node_records["signed_score"].extend(signed_score.cpu().tolist())
            node_records["contribution"].extend(contribution.cpu().tolist())
        else:
            graph_logits = output
        bag_logits = graph_to_bag_logits(graph_logits, bag_size)
        bag_labels.extend(labels.cpu().tolist())
        bag_logits_all.extend(bag_logits.cpu().tolist())
        graph_labels.extend(graph_batch.y.cpu().tolist())
        graph_logits_all.extend(graph_logits.cpu().tolist())

    bag_probabilities = 1 / (1 + np.exp(-np.asarray(bag_logits_all)))
    graph_probabilities = 1 / (1 + np.exp(-np.asarray(graph_logits_all)))
    return (
        binary_metrics(np.asarray(bag_labels), bag_probabilities),
        binary_metrics(np.asarray(graph_labels), graph_probabilities),
        node_records,
        reconstruction_error,
    )


def summarize_explanations(records: dict[str, list[float]] | None) -> dict[str, object]:
    assert records is not None
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for locus, label, attention, signed_score, contribution in zip(
        records["locus_id"],
        records["label"],
        records["attention"],
        records["signed_score"],
        records["contribution"],
    ):
        values = grouped[(int(locus), int(label))]
        values["attention"].append(float(attention))
        values["signed_score"].append(float(signed_score))
        values["contribution"].append(float(contribution))

    rows = []
    for locus in range(LOCUS_COUNT):
        per_class = {}
        for label in (0, 1):
            values = grouped[(locus, label)]
            per_class[label] = {
                "observations": len(values["contribution"]),
                "mean_attention": mean(values["attention"]),
                "mean_signed_score": mean(values["signed_score"]),
                "mean_contribution": mean(values["contribution"]),
            }
        rows.append(
            {
                "canonical_index_0_based": locus,
                "original_readout": locus + 4,
                "chrom_start": LOCI[locus],
                "paper_cbs": locus + 4 in PAPER_CBS_READOUTS,
                "untreated": per_class[0],
                "ctcf_depleted": per_class[1],
                "attention_all": mean(
                    [per_class[0]["mean_attention"], per_class[1]["mean_attention"]]
                ),
                "depleted_minus_untreated_contribution": (
                    per_class[1]["mean_contribution"]
                    - per_class[0]["mean_contribution"]
                ),
            }
        )

    cbs = [row for row in rows if row["paper_cbs"]]
    non_cbs = [row for row in rows if not row["paper_cbs"]]
    return {
        "positive_logit_class": "ctcf_depleted",
        "top_mean_attention": sorted(
            rows, key=lambda row: row["attention_all"], reverse=True
        )[:10],
        "top_differential_contribution_toward_depleted": sorted(
            rows,
            key=lambda row: row["depleted_minus_untreated_contribution"],
            reverse=True,
        )[:10],
        "top_differential_contribution_toward_untreated": sorted(
            rows, key=lambda row: row["depleted_minus_untreated_contribution"]
        )[:10],
        "paper_cbs_summary": {
            "mean_attention_cbs": mean(row["attention_all"] for row in cbs),
            "mean_attention_non_cbs": mean(row["attention_all"] for row in non_cbs),
            "mean_abs_differential_contribution_cbs": mean(
                abs(row["depleted_minus_untreated_contribution"]) for row in cbs
            ),
            "mean_abs_differential_contribution_non_cbs": mean(
                abs(row["depleted_minus_untreated_contribution"]) for row in non_cbs
            ),
        },
        "all_loci": rows,
    }


def train_one_seed(
    dataset: list[Data],
    train_indices: list[int],
    validation_bags: list[list[int]],
    test_bags: list[list[int]],
    seed: int,
    epochs: int,
    bag_size: int,
    bags_per_batch: int,
    hidden_dim: int,
    patience: int,
) -> tuple[dict[str, object], InterpretableBagGNN]:
    set_seed(seed)
    model = InterpretableBagGNN(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_validation_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_bags = make_balanced_bags(
            dataset, train_indices, bag_size, seed * 10_000 + epoch
        )
        total_loss = 0.0
        total_bags = 0
        for graph_batch, labels, selected in iter_bag_minibatches(
            dataset,
            train_bags,
            bags_per_batch,
            True,
            seed * 100_000 + epoch,
        ):
            optimizer.zero_grad()
            graph_logits = model(
                graph_batch.x, graph_batch.edge_index, graph_batch.batch
            )
            bag_logits = graph_to_bag_logits(graph_logits, bag_size)
            loss = criterion(bag_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)
            total_bags += len(selected)

        validation_metrics, _, _, _ = evaluate(
            model, dataset, validation_bags, bag_size, bags_per_batch
        )
        record = {
            "epoch": epoch,
            "training_bags": total_bags,
            "train_loss": total_loss / total_bags,
            "validation_roc_auc": validation_metrics["roc_auc"],
        }
        history.append(record)
        print(
            f"seed={seed} epoch={epoch:03d} loss={record['train_loss']:.4f} "
            f"val_auc={record['validation_roc_auc']:.4f}",
            flush=True,
        )

        if validation_metrics["roc_auc"] > best_validation_auc + 1e-4:
            best_validation_auc = validation_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_bag, validation_graph, _, _ = evaluate(
        model, dataset, validation_bags, bag_size, bags_per_batch
    )
    test_bag, test_graph, node_records, reconstruction_error = evaluate(
        model,
        dataset,
        test_bags,
        bag_size,
        bags_per_batch,
        collect_nodes=True,
    )
    return (
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_bag_metrics": validation_bag,
            "validation_graph_metrics": validation_graph,
            "test_bag_metrics": test_bag,
            "test_graph_metrics": test_graph,
            "maximum_additive_readout_reconstruction_error": reconstruction_error,
            "history": history,
            "interpretation": summarize_explanations(node_records),
        },
        model,
    )


def aggregate_seed_metrics(
    seed_results: list[dict[str, object]], key: str
) -> dict[str, object]:
    output = {}
    for metric in ("roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1"):
        values = [float(result[key][metric]) for result in seed_results]
        output[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ),
            "values": values,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--bag-size", type=int, default=10)
    parser.add_argument("--bags-per-batch", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=224)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "full_results.json",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models"
    )
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, provenance = load_dataset(args.data_dir)
    train_indices, validation_indices, test_indices = make_split(
        dataset, args.split_seed
    )
    validation_bags = make_balanced_bags(
        dataset, validation_indices, args.bag_size, args.split_seed + 1
    )
    test_bags = make_balanced_bags(
        dataset, test_indices, args.bag_size, args.split_seed + 2
    )
    reference_train_bags = make_balanced_bags(
        dataset, train_indices, args.bag_size, args.split_seed
    )
    print(
        f"Loaded {len(dataset)} graphs; trace split "
        f"{len(train_indices)}/{len(validation_indices)}/{len(test_indices)}; "
        f"fixed bags {len(reference_train_bags)}/{len(validation_bags)}/{len(test_bags)}",
        flush=True,
    )

    seed_results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_checkpoints:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        result, model = train_one_seed(
            dataset,
            train_indices,
            validation_bags,
            test_bags,
            seed,
            args.epochs,
            args.bag_size,
            args.bags_per_batch,
            args.hidden_dim,
            args.patience,
        )
        seed_results.append(result)
        if not args.no_checkpoints:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "node_features": list(NODE_FEATURE_NAMES),
                    "edge_features": [],
                    "hidden_dim": args.hidden_dim,
                    "bag_size": args.bag_size,
                    "seed": seed,
                },
                args.checkpoint_dir / f"interpretable_bag_gnn_seed_{seed}.pt",
            )

    report = {
        "status": "preliminary_random_trace_split_missingness_not_yet_controlled",
        "dataset": {
            "graphs": len(dataset),
            "minimum_observed_nodes": MIN_NODES,
            "canonical_loci": LOCUS_COUNT,
            "contact_threshold_nm": CONTACT_THRESHOLD_NM,
            "genomically_adjacent_edges_excluded": True,
            "paper_cbs_original_readouts": sorted(PAPER_CBS_READOUTS),
            "provenance": provenance,
        },
        "features": {
            "node": list(NODE_FEATURE_NAMES),
            "edge": [],
            "missing_nodes": "omitted from each variable-size graph",
        },
        "split": {
            "type": "random_70_15_15_trace_split_stratified_by_condition_and_run_group",
            "seed": args.split_seed,
            "split_before_bagging": True,
            "bags_disjoint_within_fixed_validation_and_test_partitions": True,
            "train": bag_summary(dataset, reference_train_bags),
            "validation": bag_summary(dataset, validation_bags),
            "test": bag_summary(dataset, test_bags),
        },
        "model": {
            "message_passing_layers": 1,
            "aggregation": "sum",
            "hidden_dim": args.hidden_dim,
            "edge_features": False,
            "virtual_node_during_message_passing": False,
            "graph_readout": "softmax node attention times signed node score",
            "bag_readout": "mean of graph logits",
            "bag_size": args.bag_size,
            "positive_logit_class": "ctcf_depleted",
        },
        "limitations": [
            "Locus-specific missingness is not controlled in this preliminary run.",
            "Run groups are stratified across partitions rather than held out.",
            "Multiple traces from one biological cell may not be group-split when cell identifiers are unavailable.",
        ],
        "aggregate_test_bag_metrics_across_seeds": aggregate_seed_metrics(
            seed_results, "test_bag_metrics"
        ),
        "aggregate_test_graph_metrics_across_seeds": aggregate_seed_metrics(
            seed_results, "test_graph_metrics"
        ),
        "seeds": seed_results,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
