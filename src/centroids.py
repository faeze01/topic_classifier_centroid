"""Build one centroid vector per topic by averaging embeddings of posts labeled with that topic."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import CENTROIDS_PATH, DATA_PATH
from src.embeddings import get_embedding
from src.post_text import build_embed_text

PROGRESS_EVERY = 100
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
CHECKPOINT_PATH = CENTROIDS_PATH.parent / "_embeddings_checkpoint.npy"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def safe_print(*args, **kwargs) -> None:
    """Print without crashing on Windows consoles that cannot encode Persian text."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), **kwargs)


def raise_field_size_limit() -> None:
    """Raise csv field size limit (needed for large HTML body fields on python engine)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)
            if limit < 128 * 1024:
                csv.field_size_limit(10**7)
                return


def load_posts(path: Path) -> pd.DataFrame:
    """Load the posts CSV, skipping malformed lines."""
    raise_field_size_limit()
    return pd.read_csv(
        path,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
    )


def deduplicate_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate post_id rows (topics is identical across duplicates)."""
    return df.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)


def parse_topics(topics_value: object) -> list[str]:
    """Split a comma-separated topics string into a clean list of names."""
    if topics_value is None or (isinstance(topics_value, float) and pd.isna(topics_value)):
        return []
    return [t.strip() for t in str(topics_value).split(",") if t.strip()]


def get_embedding_with_retry(text: str) -> np.ndarray:
    """Call get_embedding, retrying on transient Ollama/network errors.

    If the model reports the input still exceeds its context length, truncate
    further and retry immediately (no delay needed for this case).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return get_embedding(text)
        except Exception as exc:  # noqa: BLE001 - network/server errors from Ollama
            if attempt == MAX_RETRIES:
                raise
            if "context length" in str(exc).lower():
                text = text[: len(text) // 2]
                safe_print(f"  retry {attempt}/{MAX_RETRIES} after context-length error; truncated to {len(text)} chars")
            else:
                safe_print(f"  retry {attempt}/{MAX_RETRIES} after error: {exc}")
                time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("unreachable")  # for type checkers; loop always returns or raises


def save_checkpoint(embeddings: list[np.ndarray], path: Path) -> None:
    """Persist embeddings computed so far, so a crash does not lose completed work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(embeddings, dtype=np.float32))


def load_checkpoint(path: Path) -> list[np.ndarray]:
    """Load previously checkpointed embeddings, or an empty list if none exist."""
    if not path.exists():
        return []
    return list(np.load(path))


def embed_posts(texts: list[str]) -> np.ndarray:
    """Embed each text, printing progress since Ollama calls are slow.

    Resumes from a checkpoint file if one exists (from a previous interrupted
    run over the same texts), and checkpoints progress periodically.
    """
    total = len(texts)
    embeddings = load_checkpoint(CHECKPOINT_PATH)
    start_index = len(embeddings)
    if start_index:
        safe_print(f"Resuming from checkpoint: {start_index}/{total} already embedded.")

    for i in range(start_index, total):
        embeddings.append(get_embedding_with_retry(texts[i]))
        done = i + 1
        if done % PROGRESS_EVERY == 0 or done == total:
            safe_print(f"Embedded {done}/{total}...")
            save_checkpoint(embeddings, CHECKPOINT_PATH)

    result = np.array(embeddings, dtype=np.float32)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    return result


def build_topic_index(topic_lists: list[list[str]]) -> dict[str, list[int]]:
    """Map each topic name to the list of post indices labeled with it."""
    topic_index: dict[str, list[int]] = {}
    for i, topics in enumerate(topic_lists):
        for topic in topics:
            topic_index.setdefault(topic, []).append(i)
    return topic_index


def compute_centroids(
    embeddings: np.ndarray, topic_index: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    """Compute the mean embedding (centroid) for each topic."""
    centroids: dict[str, np.ndarray] = {}
    for topic, indices in topic_index.items():
        centroids[topic] = embeddings[indices].mean(axis=0).astype(np.float32)
    return centroids


def save_centroids(centroids: dict[str, np.ndarray], path: Path) -> None:
    """Save {topic: vector} to an NPZ file as two aligned arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    topics = np.array(list(centroids.keys()), dtype=object)
    vectors = np.stack(list(centroids.values()))
    np.savez(path, topics=topics, centroids=vectors)


def load_centroids(path: Path) -> dict[str, np.ndarray]:
    """Load {topic: vector} back from an NPZ file saved by save_centroids."""
    data = np.load(path, allow_pickle=True)
    return dict(zip(data["topics"].tolist(), data["centroids"]))


def print_topic_report(topic_index: dict[str, list[int]]) -> None:
    """Print post counts per topic, sorted fewest to most."""
    safe_print()
    safe_print("Posts per topic (fewest -> most):")
    for topic, indices in sorted(topic_index.items(), key=lambda kv: len(kv[1])):
        safe_print(f"  {topic}: {len(indices)}")
    safe_print(f"Total topics: {len(topic_index)}")


def main() -> None:
    configure_stdout()

    safe_print(f"CSV path: {DATA_PATH}")
    df = load_posts(DATA_PATH)
    safe_print(f"loaded rows: {len(df)}")

    df = deduplicate_posts(df)
    safe_print(f"unique posts after dedup: {len(df)}")

    topic_lists = [parse_topics(t) for t in df["topics"]]
    texts = [build_embed_text(t, b) for t, b in zip(df["title"], df["body"])]

    safe_print(f"Embedding {len(texts)} posts via Ollama (this may take a while)...")
    embeddings = embed_posts(texts)

    topic_index = build_topic_index(topic_lists)
    centroids = compute_centroids(embeddings, topic_index)

    save_centroids(centroids, CENTROIDS_PATH)
    safe_print(f"Saved {len(centroids)} centroids to {CENTROIDS_PATH}")

    print_topic_report(topic_index)


if __name__ == "__main__":
    main()
