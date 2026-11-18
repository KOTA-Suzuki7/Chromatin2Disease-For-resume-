# Interpretable Bag-Level GNN for 3D Chromatin Traces

This repository contains my interpretable-model contribution to a CS224W team
project. I represent single-chromosome traces of the mouse chromosome 6 Hoxa
region as variable-size contact graphs and classify whether sets of traces came
from untreated or experimentally CTCF-depleted embryonic stem cells.

The model is deliberately small: one sum-aggregation message-passing layer,
attention only at the final graph readout, and an exact decomposition of every
prediction into signed locus-level contributions.

## Preliminary result

| Evaluation unit | Test ROC-AUC | Average precision | Accuracy |
| --- | ---: | ---: | ---: |
| Individual chromosome trace | 0.637 ± 0.002 | 0.622 ± 0.003 | 0.604 ± 0.008 |
| Bag of 10 same-condition traces | **0.828 ± 0.007** | **0.831 ± 0.006** | **0.722 ± 0.014** |

Values are mean ± sample standard deviation over training seeds 11, 22, and
33 on one fixed test split. These are preliminary random-split results, not a
replicate-held-out generalization estimate.

## Biological question

CTCF helps organize chromatin loops and CTCF-binding-site hubs. Hafner et al.
used Optical Reconstruction of Chromatin Architecture (ORCA) to measure 3D
chromatin folding before and after acute CTCF depletion. Individual traces are
highly variable, so this project asks whether a GNN can detect structural
changes that become consistent when several traces from one condition are
considered together.

The two labels are:

- `0`: untreated CTCF-AID mouse embryonic stem cells;
- `1`: the same engineered cell line after four hours of auxin-induced CTCF
  depletion.

This is a perturbation-classification task, not healthy-versus-diseased
classification.

## Data

The six processed FOF-CT core tables come from two published 4DN experiment
sets:

- [Untreated chr6 traces](https://data.4dnucleome.org/experiment-set-replicates/4DNESJ3TXVIR/)
- [CTCF-depleted chr6 traces](https://data.4dnucleome.org/experiment-set-replicates/4DNES2KX6HQ5/)

After retaining traces with at least 37 of the 73 canonical loci, the dataset
contains 11,488 graphs: 5,283 untreated and 6,205 CTCF-depleted. Raw data are
not committed. See [data/README.md](data/README.md) for accessions, attribution,
and download instructions.

## Graph construction

Each observed genomic locus is a node. An undirected edge connects two
nonadjacent genomic loci when their Euclidean distance is less than 238 nm:

\[
(i,j)\in E \iff \|r_i-r_j\|_2 < 238\text{ nm}
\quad\text{and}\quad |\ell_i-\ell_j|>1.
\]

Immediately adjacent genomic segments are excluded because their proximity is
expected from polymer continuity rather than a nontrivial 3D contact.

Each node has 76 numeric inputs:

\[
x_i=[\operatorname{onehot}_{73}(\ell_i),\ c_i,\ \widetilde d_i,\ \widetilde r_i],
\]

where `c_i` is a paper-derived CTCF-binding-site indicator,
`d_i` is contact degree normalized by eligible observed partners, and `r_i` is
distance from the trace centroid normalized by radius of gyration. There are no
edge features and missing loci are omitted rather than imputed.

## Model

The single message-passing layer is

\[
h_i=\operatorname{ELU}\left(
W_{self}x_i + W_{neighbor}\sum_{j\in\mathcal N(i)}x_j+b
\right),
\]

with hidden dimension 8. Final node attention and signed scores are

\[
\alpha_i=\operatorname{softmax}_{i\in V_g}(q^Th_i+b_a),
\qquad s_i=w_s^Th_i.
\]

The graph logit is exactly

\[
z_g=b_g+\sum_{i\in V_g}\alpha_i s_i.
\]

Thus `alpha_i * s_i` is the exact signed contribution of locus `i`. Positive
contributions move the prediction toward CTCF-depleted; negative contributions
move it toward untreated. The ten-trace bag logit is

\[
z_B=\frac{1}{10}\sum_{g\in B}z_g.
\]

The model is trained with binary cross-entropy with logits.

## Reproduce the analysis

Python 3.13 was used for the reported run.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
pytest
python scripts/train.py --epochs 80 --seeds 11 22 33
```

Training writes the full metrics, histories, and locus explanations to
`results/full_results.json`. Checkpoints are written to `models/`; both are
ignored because they are reproducible local artifacts. The compact reported
metrics are tracked in [results/summary.json](results/summary.json).

Run the technical-confound audit with:

```bash
python scripts/audit_missingness.py
```

## Repository layout

```text
src/chromatin_gnn/data.py       FOF-CT loading, graph construction, splits, bags
src/chromatin_gnn/model.py      interpretable graph model and bag pooling
src/chromatin_gnn/training.py   training, metrics, checkpoints, explanations
scripts/download_data.py        reproducible 4DN downloader
scripts/train.py                training command-line entry point
scripts/audit_missingness.py    missingness-only leakage audit
tests/                          scientific and implementation invariants
results/summary.json            compact preliminary results
docs/report.md                  methods and result narrative
```

## Limitations

- Locus-specific missingness differs between conditions. Because missing nodes
  are omitted, some reported performance may reflect measurement missingness.
- The current 70/15/15 split is stratified by condition and experimental run;
  it is not run-group-held-out evaluation.
- Cell identifiers are not used for group splitting, so traces from the same
  biological cell may occur in different partitions when that relationship is
  unavailable.
- Attention identifies nodes emphasized by the learned readout, but attention
  alone is not a causal biological explanation. Signed contributions and
  targeted audits should be interpreted alongside it.

## Scope of this repository

This repository contains only my interpretable GNN, data-processing,
evaluation, and auditing contribution. It intentionally does not contain a
teammate's separate high-performance model.

## References and attribution

1. Hafner A et al. “Loop stacking organizes genome folding from TADs to
   chromosomes.” *Molecular Cell* 83, 1377–1392.e6 (2023).
   https://doi.org/10.1016/j.molcel.2023.04.008
2. Reiff SB et al. “The 4D Nucleome Data Portal as a resource for searching
   and visualizing curated nucleomics data.” *Nature Communications* 13, 2365
   (2022). https://doi.org/10.1038/s41467-022-29697-4

The MIT license applies to original code in this repository, not to the source
4DN measurements.

