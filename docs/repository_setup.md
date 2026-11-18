# Public repository setup

The local project is already initialized as a Git repository, but no commits or
remote repository have been created. Review the files and replace any desired
author metadata before publishing.

## Intended public files

```text
.github/workflows/tests.yml
.gitignore
CITATION.cff
LICENSE
README.md
pyproject.toml
requirements.txt
data/README.md
docs/report.md
docs/repository_setup.md
results/summary.json
scripts/audit_missingness.py
scripts/download_data.py
scripts/train.py
src/chromatin_gnn/__init__.py
src/chromatin_gnn/data.py
src/chromatin_gnn/model.py
src/chromatin_gnn/training.py
tests/test_graph_construction.py
tests/test_model.py
tests/test_splits_and_bags.py
```

Raw data, local virtual environments, checkpoints, full preliminary outputs,
and earlier experiments are excluded through `.gitignore`.

## Before publishing

1. Run `pytest`.
2. Run a one-epoch smoke test:

   ```bash
   python scripts/train.py --epochs 1 --patience 1 --seeds 11 \
     --no-checkpoints --output /tmp/chromatin_gnn_smoke.json
   ```

3. Review `git status --short` and confirm that only the intended files appear.
4. Stage the listed files explicitly rather than using `git add .`.
5. Create a private GitHub repository, add it as the remote, and push only
   after reviewing the rendered README.
6. Make the repository public when author metadata, attribution, and results
   are final.

Suggested repository name: `interpretable-chromatin-gnn`.

Suggested description: `Interpretable bag-level GNN for CTCF-depleted 3D chromatin traces`.
