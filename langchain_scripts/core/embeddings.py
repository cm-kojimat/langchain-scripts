from functools import lru_cache
from urllib.parse import ParseResult, parse_qs

import boto3
from langchain_community.embeddings import (
    BedrockEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)
from langchain_core.embeddings import Embeddings


@lru_cache(maxsize=8)
def detect_embedding(embedding_schema: ParseResult) -> Embeddings:
    query = parse_qs(embedding_schema.query)
    if embedding_schema.scheme == "ollama":
        return OllamaEmbeddings(model=embedding_schema.netloc)
    if embedding_schema.scheme == "openai":
        return OpenAIEmbeddings(model_name=embedding_schema.netloc)
    if embedding_schema.scheme == "bedrock":
        region = None
        if query.get("region"):
            region = query["region"][0]
        client = boto3.client("bedrock-runtime", region_name=region)
        return BedrockEmbeddings(client=client, model_id=embedding_schema.netloc)

    msg = f"Embedding {embedding_schema.geturl()} not supported."
    raise ValueError(msg)
