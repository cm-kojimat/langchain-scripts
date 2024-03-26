from langchain_core.documents import Document

from langchain_scripts.core.embed_message import embed_documents


def test_embed_documents() -> None:
    documents = [
        Document(
            page_content="# hello",
            metadata={"source": "x.md", "language": "markdown"},
        ),
    ]
    assert embed_documents(documents) == [
        {
            "type": "text",
            "text": """#source:x.md
```markdown
# hello
```""",
        },
    ]
