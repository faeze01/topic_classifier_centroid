"""Side-by-side LLM verify comparison (qwen3:4b vs gemma3:4b). Comparison-only — no backend/CSV writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.centroids import configure_stdout, deduplicate_posts, load_centroids, load_posts, safe_print  # noqa: E402
from src.config import CENTROIDS_PATH, DATA_PATH  # noqa: E402
from src.retrieve import retrieve_candidates  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from llm_verify import (  # noqa: E402
    TEST_POST_IDS,
    VERIFY_MODEL,
    VERIFY_MODEL_B,
    build_verify_messages,
    call_llm_verify,
    parse_verify_response,
    preprocess_post_text,
    print_stage1_preview,
)

COMPARE_MODELS = [VERIFY_MODEL, VERIFY_MODEL_B]


def format_model_line(model: str, result: dict) -> str:
    reason = result["reason"].replace("\n", " ")
    if result["decision"] == "match":
        topics = json.dumps(result["topics"], ensure_ascii=False)
        return f"[{model}]  decision=match  topics={topics}  reason=\"{reason}\""
    suggested = result["suggested_topic"]
    return (
        f"[{model}]  decision=no_match  suggested_topic=\"{suggested}\"  reason=\"{reason}\""
    )


def format_outcome_key(result: dict) -> str:
    if result["decision"] == "match":
        return f"match:{','.join(sorted(result['topics']))}"
    return f"no_match:{result['suggested_topic']}"


def run_model_verify(
    model: str,
    messages: list[dict[str, str]],
    candidate_topics: list[str],
) -> dict | None:
    try:
        raw = call_llm_verify(messages, model=model)
    except Exception as exc:  # noqa: BLE001
        safe_print(f"[{model}]  ERROR: {exc}")
        return None
    try:
        return parse_verify_response(raw, candidate_topics)
    except (json.JSONDecodeError, ValueError) as exc:
        safe_print(f"[{model}]  ERROR parsing response: {exc}")
        safe_print(f"[{model}]  raw: {raw}")
        return None


def compare_post(post_id: int, row, centroids: dict) -> tuple[bool, bool]:
    """Return (both_ok, models_agreed)."""
    title = row["title"]
    body = row["body"]

    safe_print()
    safe_print("=" * 72)
    safe_print(f"post_id: {post_id}")
    safe_print(f"company topics (dataset): {row.get('topics', '')}")

    candidates = retrieve_candidates(str(title), str(body), centroids, top_k=5)
    candidate_topics = [c["topic"] for c in candidates]
    print_stage1_preview(candidates)

    top2 = ", ".join(
        f"{c['topic']} ({c['similarity']:.4f})" for c in candidates[:2]
    )
    safe_print(f"Stage 1 top-2: {top2}")
    safe_print()

    messages = build_verify_messages(preprocess_post_text(title, body), candidates)
    results: dict[str, dict | None] = {}

    for model in COMPARE_MODELS:
        results[model] = run_model_verify(model, messages, candidate_topics)
        if results[model] is not None:
            safe_print(format_model_line(model, results[model]))

    both_ok = all(results[m] is not None for m in COMPARE_MODELS)
    agreed = False
    if both_ok:
        keys = [format_outcome_key(results[m]) for m in COMPARE_MODELS]
        agreed = keys[0] == keys[1]
        safe_print(f"agreement: {'YES' if agreed else 'NO'}")

    return both_ok, agreed


def main() -> None:
    configure_stdout()

    safe_print("COMPARISON ONLY — not written to backend or CSV")
    safe_print(f"Models: {VERIFY_MODEL} vs {VERIFY_MODEL_B}")
    safe_print()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)
    if not DATA_PATH.exists():
        safe_print(f"ERROR: posts file not found: {DATA_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    df = deduplicate_posts(load_posts(DATA_PATH)).set_index("post_id", drop=False)

    safe_print(f"Loaded {len(centroids)} centroids, {len(df)} posts")
    safe_print(f"Test posts: {len(TEST_POST_IDS)}")
    safe_print()

    ok_count = 0
    agree_count = 0
    missing = 0
    error_posts = 0

    for post_id in TEST_POST_IDS:
        if post_id not in df.index:
            safe_print(f"ERROR: post_id {post_id} not found in dataset")
            missing += 1
            continue

        both_ok, agreed = compare_post(post_id, df.loc[post_id], centroids)
        if both_ok:
            ok_count += 1
            if agreed:
                agree_count += 1
        else:
            error_posts += 1

    safe_print()
    safe_print("=" * 72)
    safe_print("Comparison summary")
    safe_print(f"  posts:              {len(TEST_POST_IDS)}")
    safe_print(f"  both models OK:     {ok_count}")
    safe_print(f"  models agreed:      {agree_count}")
    safe_print(f"  disagree / partial: {ok_count - agree_count}")
    safe_print(f"  model errors:       {error_posts}")
    safe_print(f"  missing post_ids:   {missing}")


if __name__ == "__main__":
    main()
