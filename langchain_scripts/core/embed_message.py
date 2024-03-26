import logging
from urllib.parse import urlparse

from langchain.prompts.prompt import PromptTemplate
from langchain_core.documents import Document
from langchain_core.prompts import format_document
from langchain_core.utils.image import image_to_data_url

logger = logging.getLogger(__name__)

SOURCE_DOCUMENT_PROMPT = PromptTemplate.from_template(
    """#source:{source}
```
{page_content}
```""",
)
CODE_DOCUMENT_PROMPT = PromptTemplate.from_template(
    """#source:{source}
```{language}
{page_content}
```""",
)


def embed_documents(documents: list[Document]) -> list[dict]:
    return [
        {
            "type": "text",
            "text": (
                format_document(doc, CODE_DOCUMENT_PROMPT)
                if doc.metadata.get("language") and doc.metadata.get("source")
                else format_document(doc, SOURCE_DOCUMENT_PROMPT)
            ),
        }
        for doc in documents
    ]


def embed_image_from_text(input_text: str) -> list[dict]:
    contents = []
    for raw_input in input_text.split("\n" * 3):
        input_stripped = raw_input.strip()
        if (
            input_stripped not in (" ", "\n")
            and input_stripped.startswith("<<")
            and input_stripped.endswith(">>")
        ):
            input_schema = urlparse(raw_input[2:-2])
            if input_schema.scheme == "image":
                data_url = image_to_data_url(input_schema.netloc + input_schema.path)
                contents.append({"type": "image_url", "image_url": {"url": data_url}})
                continue
        elif raw_input:
            contents.append({"type": "text", "text": raw_input})
    return contents
