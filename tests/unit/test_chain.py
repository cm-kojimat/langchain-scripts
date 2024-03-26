from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_scripts.chain import _combine_message

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
    "langchain_scripts.core.embed_message.image_to_data_url",
    Mock(return_value="data:image/png;base64,..."),
)
def test_combine_message(context: dict, expected_messages: list[BaseMessage]) -> None:
    messages = _combine_message(context)
    assert len(messages) == len(expected_messages)
    for message, expected_message in zip(messages, expected_messages, strict=False):
        assert type(message) == type(expected_message)
        assert message.content == expected_message.content
