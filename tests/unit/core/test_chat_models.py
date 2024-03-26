from unittest.mock import Mock, patch
from urllib.parse import urlparse

import pytest
from langchain_community.chat_models import BedrockChat, ChatOllama, ChatOpenAI

from langchain_scripts.core.chat_models import detect_chat_model


@pytest.mark.parametrize(
    ("model_schema", "expected_model"),
    [
        ("ollama://some_model", ChatOllama),
        ("openai://gpt-3.5-turbo", ChatOpenAI),
        ("bedrock://some_model_id?region=us-east-1", BedrockChat),
    ],
)
@patch("boto3.client", Mock())
@patch.dict("os.environ", {"OPENAI_API_KEY": "XXX"})
def testdetect_chat_model(model_schema: str, expected_model: type) -> None:
    chat_model = detect_chat_model(urlparse(url=model_schema))
    assert isinstance(chat_model, expected_model)
