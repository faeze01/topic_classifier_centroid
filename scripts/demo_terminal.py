"""Interactive terminal demo: title + body until 'end', then retrieve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.centroids import configure_stdout, load_centroids, safe_print  # noqa: E402
from src.config import CENTROIDS_PATH, EMBEDDING_MODEL  # noqa: E402
from src.retrieve import retrieve_candidates, topics_from_scores  # noqa: E402


def print_stage1_preview(candidates: list[dict]) -> None:
    safe_print("Stage 1 top-5:")
    for rank, row in enumerate(candidates, start=1):
        safe_print(
            f"  {rank}. {row['topic']:<24} "
            f"similarity={row['similarity']:.4f}  confidence={row['confidence']}"
        )


def read_title() -> str | None:
    try:
        title = input("عنوان: ")
    except (EOFError, KeyboardInterrupt):
        safe_print()
        return None
    return title.strip()


def read_body() -> str | None:
    safe_print("بدنه (خط جدا با end تمام):")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            safe_print()
            return None
        if line.strip() == "end":
            break
        lines.append(line)
    return "\n".join(lines)


def classify_once(title: str, body: str, centroids: dict) -> None:
    candidates = retrieve_candidates(title, body, centroids, top_k=5)
    topic_names = topics_from_scores(candidates)

    safe_print()
    print_stage1_preview(candidates)
    safe_print(f"topics={json.dumps(topic_names, ensure_ascii=False)}")


def main() -> None:
    configure_stdout()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    safe_print(f"Loaded {len(centroids)} centroids from {CENTROIDS_PATH}")
    safe_print(f"Embed: {EMBEDDING_MODEL}")
    safe_print("عنوان خالی = خروج. Ctrl+C هم خروج.")
    safe_print()

    while True:
        title = read_title()
        if title is None or title == "":
            break

        body = read_body()
        if body is None:
            break

        classify_once(title, body, centroids)
        safe_print()


if __name__ == "__main__":
    main()
