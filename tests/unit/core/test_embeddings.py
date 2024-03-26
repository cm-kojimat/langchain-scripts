from unittest.mock import Mock, patch
from urllib.parse import urlparse

import pytest
from langchain_community.embeddings import (
    BedrockEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)

from langchain_scripts.core.embeddings import detect_embedding


@pytest.mark.parametrize(
    ("embedding_schema", "expected_embedding"),
    [
        ("ollama://some_model", OllamaEmbeddings),
        ("openai://text-embedding-ada-002", OpenAIEmbeddings),
        ("bedrock://some_model_id?region=us-west-2", BedrockEmbeddings),
    ],
)
@patch("boto3.client", Mock())
@patch.dict("os.environ", {"OPENAI_API_KEY": "XXX"})
def test_detect_embedding(embedding_schema: str, expected_embedding: type) -> None:
    embedding = detect_embedding(urlparse(embedding_schema))
    assert isinstance(embedding, expected_embedding)
