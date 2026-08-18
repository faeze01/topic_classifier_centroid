"""Stage 2: LLM verification of centroid-retrieved topic candidates (read-only w.r.t. src/)."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import ollama
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.centroids import (  # noqa: E402
    configure_stdout,
    deduplicate_posts,
    load_centroids,
    load_posts,
    safe_print,
)
from src.config import CENTROIDS_PATH, DATA_PATH, ROOT  # noqa: E402
from src.preprocess import clean_text  # noqa: E402
from src.retrieve import retrieve_candidates  # noqa: E402

VERIFY_MODEL = "qwen3:4b"
VERIFY_MODEL_B = "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"
NO_MATCH_QUEUE_PATH = ROOT / "data" / "llm_no_match_queue.csv"
SNIPPET_LEN = 300

# Curated test set (~18 posts): baseline sample + category picks (see plan).
TEST_POST_IDS: list[int] = [
    # Baseline: first 10 of df.sample(40, random_state=42)
    105322,
    717082,
    244002,
    1095709,
    456399,
    482297,
    580788,
    1027172,
    34388,
    745308,
    # حقوقی
    229309,
    # بازی رایانه‌ای
    79399,
    1030140,
    # Promotional / product
    153615,
    421088,
    # ویرگول meta
    2857,
    6721,
    # Diary / narrative
    2036,
    3651,
]
TEST_POST_IDS = TEST_POST_IDS[:3]

NO_MATCH_CSV_FIELDS = [
    "post_id",
    "post_text_snippet",
    "centroid_top5",
    "llm_suggested_topic",
    "timestamp",
]

_client = ollama.Client(host=OLLAMA_HOST)


def preprocess_post_text(title: object, body: object) -> str:
    """Apply production clean_text to title and body, then join for the LLM prompt."""
    title_clean = clean_text(str(title)) if pd.notna(title) else ""
    body_clean = clean_text(str(body)) if pd.notna(body) else ""
    if title_clean and body_clean:
        return f"{title_clean}\n\n{body_clean}"
    return f"{title_clean}{body_clean}".strip()


def make_snippet(cleaned_text: str, max_len: int = SNIPPET_LEN) -> str:
    if len(cleaned_text) <= max_len:
        return cleaned_text
    return f"{cleaned_text[: max_len - 3]}... [truncated]"


def format_candidates_block(candidates: list[dict]) -> str:
    lines = []
    for rank, row in enumerate(candidates, start=1):
        lines.append(
            f"{rank}. {row['topic']} — similarity={row['similarity']:.4f} ({row['confidence']})"
        )
    return "\n".join(lines)


def build_verify_messages(cleaned_text: str, candidates: list[dict]) -> list[dict[str, str]]:
    topics_block = format_candidates_block(candidates)
    system = """شما یک طبقه‌بند موضوعی برای پست‌های فارسی ویرگول هستید.
فقط و فقط با JSON معتبر پاسخ دهید؛ هیچ متن اضافه‌ای ننویسید.

ساختار خروجی (الزامی):
- همیشه کلید decision با مقدار "match" یا "no_match" باشد.
- همیشه کلید reason (یک رشته فارسی، ۱ تا ۳ جمله) — توضیح کوتاه دلیل انتخاب.
- اگر decision="match": کلید topics (لیست رشته) — بدون suggested_topic.
- اگر decision="no_match": کلید suggested_topic (یک رشته) — بدون topics.
- از کلیدهای دیگر مانند topic، label، category و غیره استفاده نکنید.

فیلد reason باید:
- هستهٔ موضوع پست را در یک خط بیان کند.
- توضیح دهد چرا topic(های) انتخاب‌شده مناسب‌اند، یا چرا candidates رد شدند.
- اگر candidateهای رتبه‌بالاتر را رد کردید، نام ببرید و بگویید چرا دقیق‌تر نیستند.

دقت انتخاب:
- فقط موضوع(های)ی را انتخاب کنید که به‌طور مشخص و مستقیم هستهٔ موضوع پست را توصیف می‌کنند.
- اگر موضوعی دقیق‌تر در فهرست candidates هست، موضوع پهن‌تر یا tangentially مرتبط را انتخاب نکنید.
- اگر چند موضوع هم‌سطح و دقیق مناسب‌اند، همهٔ آن‌ها را در topics بنویسید.

مثال ۱ (match — انتخاب دقیق‌ترین موضوع):
پست: «راهنمای حقوقی برای قرارداد کار و حقوق شغلی کارگران»
Candidates:
1. حقوقی — similarity=0.72 (strong)
2. شغل و کار — similarity=0.65 (strong)
Output: {"decision": "match", "topics": ["حقوقی"], "reason": "پست مستقیماً درباره حقوق قرارداد کار است؛ حقوقی دقیق‌تر از شغل و کار است."}

مثال ۲ (no_match — تبلیغ محصول):
پست: «معرفی پارتیشن اداری چوبی و مزایای خرید آن برای دکوراسیون شرکت»
Candidates:
1. طراحی دیجیتال — similarity=0.55 (weak)
2. بهره وری — similarity=0.52 (weak)
Output: {"decision": "no_match", "suggested_topic": "تبلیغات و معرفی محصول", "reason": "پست تبلیغ محصول پارتیشن است؛ هیچ candidate موضوع تبلیغاتی را پوشش نمی‌دهد."}"""
    user = f"""متن پست (پس از پیش‌پردازش):

{cleaned_text}

پنج موضوع پیشنهادی مرحله بازیابی (centroid) با رتبه و امتیاز:
{topics_block}

وظیفه:
- اگر یک یا چند مورد از این پنج موضوع واقعاً و به‌طور مشخص با محتوای پست هم‌خوانی دارد، decision را "match" قرار دهید و فقط همان موضوع(های) دقیق را در topics بنویسید.
- اگر هیچ‌کدام مناسب نیست، decision را "no_match" قرار دهید و suggested_topic پیشنهاد دهید.
- در reason توضیح دهید چرا این انتخاب را کردید (و اگر candidate رتبه‌بالاتری را رد کردید، چرا).

قوانین:
- topics فقط می‌تواند زیرمجموعه‌ای از پنج نام بالا باشد.
- در حالت no_match فیلد topics نباید وجود داشته باشد.
- در حالت match فیلد suggested_topic نباید وجود داشته باشد.
- کلیدهای مجاز: decision + reason + topics (لیست) یا decision + reason + suggested_topic (رشته).

فرمت خروجی:
{{"decision": "match", "topics": ["نام موضوع"], "reason": "توضیح کوتاه"}}
{{"decision": "no_match", "suggested_topic": "نام پیشنهادی", "reason": "توضیح کوتاه"}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm_verify(messages: list[dict[str, str]], model: str | None = None) -> str:
    response = _client.chat(model=model or VERIFY_MODEL, messages=messages, format="json")
    return response.message.content or ""


def parse_verify_response(raw: str, candidate_topics: list[str]) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")

    decision = data.get("decision")
    if decision not in {"match", "no_match"}:
        raise ValueError(f"invalid decision: {decision!r}")

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("response requires non-empty reason string")
    reason = reason.strip()

    allowed = set(candidate_topics)

    if decision == "match":
        topics = data.get("topics")
        if not isinstance(topics, list) or not topics:
            raise ValueError("match requires non-empty topics list")
        topics = [str(t).strip() for t in topics if str(t).strip()]
        if not topics:
            raise ValueError("match topics list is empty after normalization")
        invalid = [t for t in topics if t not in allowed]
        if invalid:
            raise ValueError(f"topics not in candidates: {invalid}")
        return {"decision": "match", "topics": topics, "reason": reason}

    suggested = data.get("suggested_topic")
    if not isinstance(suggested, str) or not suggested.strip():
        raise ValueError("no_match requires non-empty suggested_topic")
    return {
        "decision": "no_match",
        "suggested_topic": suggested.strip(),
        "reason": reason,
    }


def append_no_match_record(
    *,
    post_id: int,
    post_text_snippet: str,
    centroid_top5: str,
    llm_suggested_topic: str,
) -> None:
    NO_MATCH_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not NO_MATCH_QUEUE_PATH.exists() or NO_MATCH_QUEUE_PATH.stat().st_size == 0
    timestamp = datetime.now(timezone.utc).isoformat()

    with NO_MATCH_QUEUE_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=NO_MATCH_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "post_id": post_id,
                "post_text_snippet": post_text_snippet,
                "centroid_top5": centroid_top5,
                "llm_suggested_topic": llm_suggested_topic,
                "timestamp": timestamp,
            }
        )


def print_stage1_preview(candidates: list[dict]) -> None:
    safe_print("Stage 1 top-5:")
    for rank, row in enumerate(candidates, start=1):
        safe_print(
            f"  {rank}. {row['topic']:<24} "
            f"similarity={row['similarity']:.4f}  confidence={row['confidence']}"
        )


def process_post(post_id: int, row: pd.Series, centroids: dict) -> str | None:
    """Process one post. Returns decision label or None on failure."""
    title = row["title"]
    body = row["body"]
    cleaned_text = preprocess_post_text(title, body)

    safe_print()
    safe_print("=" * 72)
    safe_print(f"post_id: {post_id}")
    safe_print(f"company topics (dataset): {row.get('topics', '')}")

    candidates = retrieve_candidates(str(title), str(body), centroids, top_k=5)
    candidate_topics = [c["topic"] for c in candidates]
    print_stage1_preview(candidates)

    messages = build_verify_messages(cleaned_text, candidates)
    try:
        raw = call_llm_verify(messages)
    except Exception as exc:  # noqa: BLE001 — continue to next post on Ollama failure
        safe_print(f"ERROR calling LLM for post_id={post_id}: {exc}")
        return None

    safe_print(f"LLM raw: {raw}")

    try:
        result = parse_verify_response(raw, candidate_topics)
    except (json.JSONDecodeError, ValueError) as exc:
        safe_print(f"ERROR parsing LLM response for post_id={post_id}: {exc}")
        return None

    safe_print(f"LLM reason: {result['reason']}")

    centroid_json = json.dumps(candidates, ensure_ascii=False)

    if result["decision"] == "match":
        payload = {"post_id": post_id, "topics": result["topics"]}
        safe_print(f"BACKEND WRITE: {json.dumps(payload, ensure_ascii=False)}")
        return "match"

    payload = {"post_id": post_id, "topics": []}
    safe_print(f"BACKEND WRITE: {json.dumps(payload, ensure_ascii=False)}")
    safe_print("(no topic assigned — queued for human review)")

    append_no_match_record(
        post_id=post_id,
        post_text_snippet=make_snippet(cleaned_text),
        centroid_top5=centroid_json,
        llm_suggested_topic=result["suggested_topic"],
    )
    safe_print(f"Appended to {NO_MATCH_QUEUE_PATH}")
    return "no_match"


def main() -> None:
    configure_stdout()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)
    if not DATA_PATH.exists():
        safe_print(f"ERROR: posts file not found: {DATA_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    df = deduplicate_posts(load_posts(DATA_PATH)).set_index("post_id", drop=False)

    safe_print(f"Loaded {len(centroids)} centroids, {len(df)} posts")
    safe_print(f"Verify model: {VERIFY_MODEL}")
    safe_print(f"Test posts: {len(TEST_POST_IDS)}")

    counts = {"match": 0, "no_match": 0, "error": 0, "missing": 0}

    for post_id in TEST_POST_IDS:
        if post_id not in df.index:
            safe_print(f"ERROR: post_id {post_id} not found in dataset")
            counts["missing"] += 1
            continue

        decision = process_post(post_id, df.loc[post_id], centroids)
        if decision == "match":
            counts["match"] += 1
        elif decision == "no_match":
            counts["no_match"] += 1
        else:
            counts["error"] += 1

    safe_print()
    safe_print("=" * 72)
    safe_print("Summary")
    safe_print(f"  match:    {counts['match']}")
    safe_print(f"  no_match: {counts['no_match']}")
    safe_print(f"  error:    {counts['error']}")
    safe_print(f"  missing:  {counts['missing']}")
    if counts["no_match"]:
        safe_print(f"  queue:    {NO_MATCH_QUEUE_PATH}")


if __name__ == "__main__":
    main()
