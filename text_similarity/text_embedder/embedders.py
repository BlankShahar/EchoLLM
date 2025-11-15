from functools import lru_cache

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


@lru_cache
def openai_embedder(text: str, model='text-embedding-3-small') -> list[float]:
    openai_client = OpenAI()
    response = openai_client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


@lru_cache
def _load_sbert_model(model: str) -> SentenceTransformer:
    return SentenceTransformer(model)


@lru_cache
def sbert_embedder(
    text: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = False,
) -> list[float]:
    model = _load_sbert_model(model)
    emb = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=normalize,  # set True if you want cosine-friendly vectors
    )
    return np.asarray(emb, dtype=np.float32).tolist()
