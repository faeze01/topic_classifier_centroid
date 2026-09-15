"""Shared paths and flags for the topic-classifier project."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "posts.csv"
CENTROIDS_PATH = ROOT / "data" / "centroids.npz"
EMBEDDINGS_PATH = ROOT / "data" / "post_embeddings.npz"


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        os.environ.setdefault(key, value)


_load_env(ROOT / ".env")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").strip()
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
SECONDARY_MAX_GAP = float(os.getenv("SECONDARY_MAX_GAP", "0.008"))
