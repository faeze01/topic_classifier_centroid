"""Shared paths for the topic-classifier project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "posts.csv"
CENTROIDS_PATH = ROOT / "data" / "centroids.npz"
