import logging
from operator import itemgetter
from urllib.parse import parse_qs, urlparse

from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)

from langchain_scripts.core.chat_models import detect_chat_model
from langchain_scripts.core.embed_message import embed_documents, embed_image_from_text
from langchain_scripts.core.vector_store import build_retriever, get_vectorstore

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


def _combine_message(context: dict) -> list[BaseMessage]:
    messages = []
    system_prompt = context.get("system")
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.extend(context.get("chat_history", []))

    human_contents = []
    human_contents.extend(embed_image_from_text(input_text=context.get("input", "")))
    human_contents.extend(embed_documents(context.get("documents", [])))
    if human_contents:
        messages.append(HumanMessage(content=human_contents))
    return messages


def _load_vectorstores(context: dict) -> list[FAISS]:
    if not context.get("input") or not context.get("embedding"):
        return context.get("vectorstores", [])

    input_text = context["input"]

    vectorstores = []
    for raw_input in input_text.split("\n" * 3):
        input_stripped = raw_input.strip()
        if (
            input_stripped not in (" ", "\n")
            and input_stripped.startswith("<<")
            and input_stripped.endswith(">>")
        ):
            input_schema = urlparse(input_stripped[2:-2])
            input_schema_query = parse_qs(input_schema.query)
            if "embedding" in input_schema_query:
                embedding_schema = urlparse(input_schema_query["embedding"][0])
            else:
                embedding_schema = urlparse(context["embedding"])
            vectorstore = get_vectorstore(
                embedding_schema=embedding_schema,
                input_schema=input_schema,
            )
            if vectorstore:
                vectorstores.append(vectorstore)

    return vectorstores


def _retrive_documents(context: dict) -> list[Document]:
    if (
        not context.get("vectorstores")
        or not context.get("embedding")
        or not context.get("input")
    ):
        return context.get("documents", [])

    base_vectorstore: FAISS | None = None
    for vectorstore in context["vectorstores"]:
        if base_vectorstore is None:
            base_vectorstore = vectorstore
            continue
        base_vectorstore.merge_from(vectorstore)

    if base_vectorstore is None:
        return []

    embedding_schema = urlparse(context["embedding"])
    query = parse_qs(embedding_schema.query)
    retriever = build_retriever(base_vectorstore, query)
    return retriever.invoke(input=context["input"])


def build(model: str) -> RunnableSerializable:
    model_schema = urlparse(model)
    chat_model = detect_chat_model(model_schema=model_schema)

    memory = ConversationBufferMemory(
        return_messages=True,
        output_key="answer",
        input_key="input",
    )
    memory_loader = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    def _memory_save_context(context: dict) -> dict:
        memory.save_context(inputs=context, outputs=context)
        return context

    memory_saver = RunnableLambda(_memory_save_context)

    return (
        memory_loader
        | RunnablePassthrough.assign(
            vectorstores=RunnableLambda(_load_vectorstores),
        )
        | RunnablePassthrough.assign(documents=RunnableLambda(_retrive_documents))
        | RunnablePassthrough.assign(messages=RunnableLambda(_combine_message))
        | RunnablePassthrough.assign(
            answer=itemgetter("messages") | chat_model | StrOutputParser(),
        )
        | memory_saver
    )
