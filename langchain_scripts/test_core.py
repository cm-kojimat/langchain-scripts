from unittest.mock import Mock, patch
from urllib.parse import urlparse

import pytest
from langchain_community.chat_models import BedrockChat, ChatOllama, ChatOpenAI
from langchain_community.embeddings import (
    BedrockEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_scripts.core import (
    _combine_documents,
    _combine_message,
    _detect_chat_model,
    _detect_embedding,
)


def test_detect_chat_model() -> None:
    # Test OpenAI chat model
    openai_model_schema = urlparse("openai://gpt-3.5-turbo?streaming=True")
    openai_mock = Mock(spec=ChatOpenAI)
    with patch("langchain_scripts.core.ChatOpenAI", return_value=openai_mock):
        chat_model = _detect_chat_model(openai_model_schema)
        assert isinstance(chat_model, BaseChatModel)
        assert isinstance(chat_model, ChatOpenAI)

    # Test Ollama chat model
    ollama_model_schema = urlparse("ollama://my-ollama-model")
    ollama_mock = Mock(spec=ChatOllama)
    with patch("langchain_scripts.core.ChatOllama", return_value=ollama_mock):
        chat_model = _detect_chat_model(ollama_model_schema)
        assert isinstance(chat_model, BaseChatModel)
        assert isinstance(chat_model, ChatOllama)

    bedrock_model_schema = urlparse("bedrock://my-bedrock-model?region=us-east-1")
    bedrock_mock = Mock(spec=BedrockChat)
    with patch("langchain_scripts.core.BedrockChat", return_value=bedrock_mock), patch(
        "langchain_scripts.core.boto3.client",
    ) as mock_client:
        chat_model = _detect_chat_model(bedrock_model_schema)
        assert isinstance(chat_model, BaseChatModel)
        assert isinstance(chat_model, BedrockChat)
        mock_client.assert_called_with("bedrock-runtime", region_name="us-east-1")

    unsupported_model_schema = urlparse("unsupported://my-model")
    with pytest.raises(
        ValueError,
        match="Model unsupported://my-model not supported.",
    ):
        _detect_chat_model(unsupported_model_schema)


def test_detect_embedding() -> None:
    openai_embedding_schema = urlparse("openai://text-embedding-ada-002")
    openai_mock = Mock(spec=OpenAIEmbeddings)
    with patch("langchain_scripts.core.OpenAIEmbeddings", return_value=openai_mock):
        embedding = _detect_embedding(openai_embedding_schema)
        assert isinstance(embedding, Embeddings)
        assert isinstance(embedding, OpenAIEmbeddings)

    ollama_embedding_schema = urlparse("ollama://my-ollama-model")
    ollama_mock = Mock(spec=OllamaEmbeddings)
    with patch("langchain_scripts.core.OllamaEmbeddings", return_value=ollama_mock):
        embedding = _detect_embedding(ollama_embedding_schema)
        assert isinstance(embedding, Embeddings)
        assert isinstance(embedding, OllamaEmbeddings)

    bedrock_embedding_schema = urlparse("bedrock://my-bedrock-model?region=us-east-1")
    bedrock_mock = Mock(spec=BedrockEmbeddings)
    with patch(
        "langchain_scripts.core.BedrockEmbeddings",
        return_value=bedrock_mock,
    ), patch(
        "langchain_scripts.core.boto3.client",
    ) as mock_client:
        embedding = _detect_embedding(bedrock_embedding_schema)
        assert isinstance(embedding, Embeddings)
        assert isinstance(embedding, BedrockEmbeddings)
        mock_client.assert_called_with("bedrock-runtime", region_name="us-east-1")

    unsupported_model_schema = urlparse("unsupported://my-model")
    with pytest.raises(
        ValueError,
        match="Embedding unsupported://my-model not supported.",
    ):
        _detect_embedding(unsupported_model_schema)


_test_combine_message_args: dict[str, tuple[dict, list[BaseMessage]]] = {
    "system and documents": (
        {
            "system": "You are a helpful assistant.",
            "documents": [HumanMessage(content="This is a document.")],
        },
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="This is a document."),
        ],
    ),
    "chat_history": (
        {
            "chat_history": [
                HumanMessage(content="Hello"),
                SystemMessage(content="Hi there!"),
            ],
        },
        [
            HumanMessage(content="Hello"),
            SystemMessage(content="Hi there!"),
        ],
    ),
    "input": (
        {"input": "This is a text input."},
        [
            HumanMessage(
                content=[{"type": "text", "text": "This is a text input."}],
            ),
        ],
    ),
    "image input": (
        {"input": "<<file://example.com/image.png>>"},
        [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,..."},
                    },
                ],
            ),
        ],
    ),
    "image input with text": (
        {
            "input": "<<file://example.com/image.png>>\n\nSome text",
        },
        [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,..."},
                    },
                    {"type": "text", "text": "Some text"},
                ],
            ),
        ],
    ),
}


@pytest.mark.parametrize(
    argnames=("context", "expected_messages"),
    argvalues=_test_combine_message_args.values(),
    ids=_test_combine_message_args.keys(),
)
@patch(
    "langchain_scripts.core.image_to_data_url",
    Mock(return_value="data:image/png;base64,..."),
)
def test_combine_message(context: dict, expected_messages: list[BaseMessage]) -> None:
    messages = _combine_message(context)
    assert len(messages) == len(expected_messages)
    for message, expected_message in zip(messages, expected_messages, strict=False):
        assert type(message) == type(expected_message)
        assert message.content == expected_message.content


def test_combine_documents() -> None:
    documents = [
        Document(
            page_content="# hello",
            metadata={"source": "x.md", "language": "markdown"},
        ),
    ]
    assert _combine_documents(documents) == [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """#source:x.md
```markdown
# hello
```""",
                },
            ],
        ),
    ]
