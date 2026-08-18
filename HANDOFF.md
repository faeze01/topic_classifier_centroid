I'm continuing a Persian-language multi-label topic classification project. Here's the full context so you can pick up exactly where I left off:

GOAL:

Build a system that assigns 1+ topics (out of 50 predefined categories) to Persian blog posts on a platform called Virgool, at publish time. The existing dataset used for this has noisy/unreliable labels.

CONSTRAINTS:

- Hardware: RTX 3050 GPU, must run fully offline/local (no cloud APIs) — using Ollama

- Language: Persian text, needs to handle idiom/sarcasm/cultural references (e.g. "شاگردان کی‌روش" meaning "Iran's national football team")

- I prefer understanding concepts before writing code, minimal/non-overengineered solutions, and Python with Cursor IDE

ARCHITECTURE DECIDED (retrieve → verify, two-stage):

1. Embed post text using an embedding model

2. Build one centroid vector per topic (mean of embeddings of posts already labeled with that topic) — chosen over KNN because centroids average out label noise

3. Compare new post's embedding to all 50 centroids via cosine similarity → get top 3-5 candidate topics

4. Pass full post text + tags + the candidate topic NAMES (as plain text, not vectors) to a local LLM (qwen3:4b via Ollama) to make the final multi-label decision

5. Side branch: reuse the same centroids to detect topics that should be merged (very similar centroids) or split (posts far from their own topic's centroid)

EMBEDDING MODEL DECISION:

Tested qwen3-embedding:4b vs bge-m3 vs a considered SOTA Persian-specific model called Hakim (couldn't use it — its API requires an internet connection to a server not reachable from Iran; local weights via sentence-transformers would be the workaround if revisited).

Result: BGE-M3 significantly outperformed qwen3-embedding:4b on culturally-idiomatic Persian test pairs (0.78 vs 0.45 similarity on a "same topic, cultural reference" pair) while performing similarly or better on clean paraphrase and sarcasm tests. DECISION: use bge-m3.

REJECTED APPROACH: Fine-tuning ParsBERT directly as a classifier — rejected because it would memorize the existing dataset's label noise rather than being robust to it, and because ParsBERT (like base BERT) isn't purpose-trained for embedding/similarity tasks (it's a masked-language-model, not a contrastive-trained embedding model).

BM25: Considered but decided NOT needed for now — exact-keyword matching value is already covered by giving the LLM verify step the raw text + tags directly.

PREPROCESSING (src/[preprocess.py](http://preprocess.py), function clean_text):

NFKC unicode normalization, Arabic→Persian character mapping (ي→ی, ك→ک, ة→ه, ؤ→و, ئ→ی), tatweel removal, strip invisible direction marks (preserve ZWNJ), strip Markdown/HTML, normalize whitespace. Explicitly does NOT remove stopwords/punctuation/stemming (these would hurt embedding quality, unlike classic NLP pipelines).

DATA FORMAT (incoming from backend, not yet received):

CSV, UTF-8 encoded, columns: post_id, title, body, topic(s), tag(s). ~150 posts per topic, format of multi-topic field (comma-separated vs JSON list vs one row per post-topic) not yet confirmed — need to run scripts/inspect_[data.py](http://data.py) once the file arrives.

EVALUATION PLAN:

Hold out ~100-150 posts as a manually-verified test set (separate from centroid-building data, since the main dataset is noisy). Use precision/recall/F1 per-topic AND overall — not raw accuracy, since this is multi-label.

CURRENT STATUS:

- src/[embeddings.py](http://embeddings.py) implemented and tested (get_embedding, get_embeddings via Ollama)

- src/[preprocess.py](http://preprocess.py) implemented

- Waiting for real dataset from backend

- NEXT STEP: src/[centroids.py](http://centroids.py) — build one centroid per topic from the labeled dataset once it arrives

Please continue helping me from here, keeping the same style: explain concepts before giving code, keep code minimal, and give me Cursor-ready prompts one step at a time.

---

## Topic discovery & validation phase (closure summary)

**Stage:** Closing unsupervised audit / centroid validation; moving to **Stage 2 (LLM verification)**.

### Centroid retrieval vs company labels

Manual review used a **three-way comparison** on **40 sampled posts** (company labels vs centroid top-5 vs HDBSCAN cluster context).

- Centroid predictions were **equal to or better than** original company labels in **~97%** of cases.
- **15 / 40** cases: centroid assignment was **clearly more accurate** than the company label.
- **0 / 40** cases: original company label was **clearly better** than centroid.

**Conclusion:** Centroid-based retrieval is validated for Stage 2; company labels remain noisy.

### HDBSCAN taxonomy audit (`min_cluster_size=30`, **27 clusters**, 7463 deduped posts)

| Finding | Detail |
|--------|--------|
| **Reliable taxonomy** | ~**19 clusters** have a **single dominant company topic** (>50% of posts in cluster) — taxonomy holds there. |
| **Merge / alias candidates** | Some clusters (e.g. resilience/تاب‌آوری, philosophical/spiritual, child-rearing) **split across related company topics** — taxonomy refinement, not retrieval failure. |
| **Cluster 1** (~**20%**, 1497 posts) | **Not topic-driven** — promotional / product-description style; poor fit to the 50 topics. **Open product decision** (new tag, exclude from classification, or other) — **not decided**. |
| **Cluster 26** (~**9.5%**) | Narrative / diary-style first-person content — **style-driven**, same open question as cluster 1 — **flagged, not resolved**. |
| **Cluster 25** (~**0.4%**) | Meta posts about Virgool — **style-driven** — **flagged, not resolved**. |
| **Noise** (~**31.6%**, label -1) | **Not fully characterized** — **follow-up** if clustering is revisited. |

### Workspace after cleanup

Exploratory discovery work under `topic_discovery/` was removed. **Kept:** project-level `cache/embeddings.npy` and `cache/post_ids.npy` (7463 post embeddings, aligned by `post_ids.npy`) for possible reuse without re-embedding via Ollama. No loader in `src/` yet — reload with `numpy` + `post_id` alignment when needed.

### Next step

**Stage 2:** LLM verification on the retrieve → verify pipeline (`src/`), using `data/centroids.npz` and production retrieval — not the removed discovery scripts.