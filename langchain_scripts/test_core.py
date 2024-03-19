from unittest.mock import ANY, Mock, patch
from urllib.parse import ParseResult, urlparse

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
    _get_documents,
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
            "documents": [
                Document(
                    page_content="> This is a document.",
                    metadata={"source": "x.md", "language": "markdown"},
                ),
            ],
        },
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "#source:x.md\n```markdown\n> This is a document.\n```",
                    },
                ],
            ),
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


@pytest.mark.parametrize(
    ("schema", "expected_document_args", "expected_search_args"),
    [
        (urlparse("ollama://codellama/test"), {}, {}),
        (urlparse("ollama://codellama/test?glob=*.txt"), {"glob": "*.txt"}, {}),
        (
            urlparse("ollama://codellama/test?exclude=*.txt&exclude=*.yml"),
            {"exclude": ["*.txt", "*.yml"]},
            {},
        ),
        (
            urlparse("ollama://codellama/test?show_progress=1"),
            {"show_progress": True},
            {},
        ),
        (
            urlparse("ollama://codellama/test?language=python"),
            {},
            {},
        ),
        (
            urlparse("ollama://codellama/test?suffixes=.txt&suffixes=.md"),
            {"suffixes": [".txt", ".md"]},
            {},
        ),
        (
            urlparse("ollama://codellama/test?search_type=similarity"),
            {},
            {"search_type": "similarity"},
        ),
        (
            urlparse("ollama://codellama/test?search_kwargs_k=10"),
            {},
            {"search_kwargs": {"k": 10}},
        ),
    ],
)
@patch("langchain_scripts.core.FAISS")
@patch("langchain_scripts.core.GenericLoader")
def test_get_documents(
    mock_loader: Mock,
    mock_vectorstore: Mock,
    schema: ParseResult,
    expected_document_args: dict,
    expected_search_args: dict,
) -> None:
    _get_documents(schema)

    mock_loader.from_filesystem.assert_called_once_with(
        path=schema.path,
        parser=ANY,
        **expected_document_args,
    )
    mock_vectorstore.from_documents.return_value.as_retriever.assert_called_once_with(
        **expected_search_args,
    )
