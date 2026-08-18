import numpy as np
import ollama

MODEL = "bge-m3"
HOST = "http://localhost:11434"

_client = ollama.Client(host=HOST)


def get_embedding(text: str) -> np.ndarray:
    """Return a sentence embedding vector for the given text."""
    response = _client.embed(model=MODEL, input=text)
    return np.array(response.embeddings[0], dtype=np.float32)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Return embedding vectors for a list of texts."""
    return np.array([get_embedding(text) for text in texts], dtype=np.float32)


