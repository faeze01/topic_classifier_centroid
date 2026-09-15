from __future__ import annotations

from src.centroids import load_centroids
from src.config import CENTROIDS_PATH
from src.retrieve import retrieve_candidates, topics_from_scores
from src.schemas import ClassifyResponse, TopicResult

_centroids = None


def _get_centroids() -> dict:
    global _centroids
    if _centroids is None:
        _centroids = load_centroids(CENTROIDS_PATH)
    return _centroids


def run_classification(body: str, title: str, post_id: str) -> ClassifyResponse:
    print(f"classify start post_id={post_id}", flush=True)
    print("classify retrieve...", flush=True)
    centroids = _get_centroids()
    candidates = retrieve_candidates(title, body, centroids)
    topic_names = topics_from_scores(candidates)
    print(f"classify topics={topic_names}", flush=True)

    return ClassifyResponse(
        post_id=post_id,
        topics=[TopicResult(topic=topic) for topic in topic_names],
    )
