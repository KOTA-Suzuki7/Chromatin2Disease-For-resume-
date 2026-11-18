"""Interpretable graph learning for single-chromosome chromatin traces."""

from .data import LOCUS_COUNT, NODE_FEATURE_DIM
from .model import InterpretableBagGNN

__all__ = ["InterpretableBagGNN", "LOCUS_COUNT", "NODE_FEATURE_DIM"]
