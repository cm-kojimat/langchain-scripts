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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_scripts.core import (
    _combine_documents,
    _combine_message,
    _detect_chat_model,
    _detect_embedding,
)


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
def test_detect_chat_model(model_schema: str, expected_model: type) -> None:
    chat_model = _detect_chat_model(urlparse(url=model_schema))
    assert isinstance(chat_model, expected_model)


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
    embedding = _detect_embedding(urlparse(embedding_schema))
    assert isinstance(embedding, expected_embedding)


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
        {"input": "<<image://assets/image.png>>"},
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
            "input": "<<image:/assets/image.png>>\n\n\nSome text",
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
    "file upload": (
        {
            "input": "please read pdf\n\n\n<<pdf:/assets/image.pdf>>",
        },
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "please read pdf"},
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
        {
            "type": "text",
            "text": """#source:x.md
```markdown
# hello
```""",
        },
    ]
