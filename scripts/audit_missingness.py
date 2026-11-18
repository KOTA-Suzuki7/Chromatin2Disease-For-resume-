#!/usr/bin/env python3
"""Measure how well locus-observation masks alone predict the condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chromatin_gnn.data import DEFAULT_DATA_DIR, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    dataset, _ = load_dataset(args.data_dir)
    masks = np.stack(
        [graph.observed_mask.squeeze(0).numpy().astype(np.int8) for graph in dataset]
    )
    labels = np.asarray([int(graph.y.item()) for graph in dataset])
    groups = np.asarray([int(graph.run_group.item()) for graph in dataset])

    results = []
    for held_out_group in sorted(set(groups)):
        train = groups != held_out_group
        test = groups == held_out_group
        classifier = BernoulliNB(alpha=1.0).fit(masks[train], labels[train])
        probabilities = classifier.predict_proba(masks[test])[:, 1]
        predictions = probabilities >= 0.5
        node_counts = masks[test].sum(axis=1)
        results.append(
            {
                "held_out_run_group": int(held_out_group),
                "test_graphs": int(test.sum()),
                "missingness_mask_roc_auc": float(
                    roc_auc_score(labels[test], probabilities)
                ),
                "missingness_mask_balanced_accuracy": float(
                    balanced_accuracy_score(labels[test], predictions)
                ),
                "node_count_roc_auc": float(
                    roc_auc_score(labels[test], node_counts)
                ),
            }
        )
    print(
        json.dumps(
            {
                "question": "Can observation masks alone predict CTCF depletion?",
                "method": "Bernoulli naive Bayes with leave-one-run-group-out evaluation",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
