#!/usr/bin/env python3
"""Command-line entry point for model training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chromatin_gnn.training import main


if __name__ == "__main__":
    main()
