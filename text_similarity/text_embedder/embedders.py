from functools import lru_cache

from openai import OpenAI

openai_client = OpenAI()


@lru_cache
def openai_embedder(text: str, model='text-embedding-3-small') -> list[float]:
    response = openai_client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
