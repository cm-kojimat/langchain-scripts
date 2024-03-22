import logging
from collections.abc import Iterable
from functools import lru_cache
from hashlib import sha256
from operator import itemgetter
from pathlib import Path
from urllib.parse import ParseResult, parse_qs, urlparse

import boto3
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain_community.chat_models import BedrockChat, ChatOllama, ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.embeddings import (
    BedrockEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.utils.image import image_to_data_url
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

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


@lru_cache(maxsize=8)
def _detect_chat_model(model_schema: ParseResult) -> BaseChatModel:
    query = parse_qs(model_schema.query)
    if model_schema.scheme == "ollama":
        return ChatOllama(model=model_schema.netloc)
    if model_schema.scheme == "openai":
        return ChatOpenAI(model_name=model_schema.netloc, streaming=True)
    if model_schema.scheme == "bedrock":
        region = None
        if query.get("region"):
            region = query["region"][0]
        client = boto3.client("bedrock-runtime", region_name=region)
        return BedrockChat(client=client, model_id=model_schema.netloc, streaming=True)

    msg = f"Model {model_schema.geturl()} not supported."
    raise ValueError(msg)


@lru_cache(maxsize=8)
def _detect_embedding(embedding_schema: ParseResult) -> Embeddings:
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


def _is_existing_faiss_vectorstore(
    folder_dir: Path,
    index_name: str = "index",
) -> bool:
    return Path(folder_dir).joinpath(f"{index_name}.faiss").exists()


def _get_file_documents(path: str, query: dict) -> Iterable[Document]:
    document_args = {}
    if query.get("glob"):
        document_args["glob"] = query["glob"][0]
    if query.get("exclude"):
        document_args["exclude"] = query["exclude"]
    if query.get("show_progress"):
        document_args["show_progress"] = bool(query["show_progress"][0])
    if query.get("suffixes"):
        document_args["suffixes"] = query["suffixes"]

    logger.info("documents args: %s", document_args)
    loader = GenericLoader.from_filesystem(
        path=path,
        parser=LanguageParser(),
        **document_args,
    )
    documents = loader.load()
    if query.get("language"):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language(query["language"][0]),
        )
        documents = splitter.split_documents(documents=documents)
    yield from documents


def _get_vectorstore(
    embedding_schema: ParseResult,
    input_schema: ParseResult,
) -> FAISS | None:
    query = parse_qs(embedding_schema.query)
    embedding = _detect_embedding(embedding_schema=embedding_schema)

    if query.get("faiss_folder"):
        folder_dir = Path(query["faiss_folder"][0])
    else:
        folder_dir = Path.joinpath(
            Path.home(),
            ".cache",
            "faiss",
            sha256(embedding_schema.geturl().encode()).hexdigest(),
        )
    index_name = sha256(input_schema.geturl().encode()).hexdigest()
    if _is_existing_faiss_vectorstore(folder_dir=folder_dir, index_name=index_name):
        return FAISS.load_local(
            folder_path=str(folder_dir),
            index_name=index_name,
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )

    documents = list(_get_documents_from_input(input_schema=input_schema))
    if not documents:
        return None

    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)
    vectorstore.save_local(folder_path=str(folder_dir), index_name=index_name)
    return vectorstore


def _build_retriever(
    vectorstore: FAISS,
    query: dict,
) -> VectorStoreRetriever:
    retriever_args = {}
    if query.get("search_type"):
        retriever_args["search_type"] = query["search_type"][0]
    for k, v in [(k, v) for k, v in query.items() if k.startswith("search_kwargs_")]:
        kwargs_name = k[len("search_kwargs_") :]
        if kwargs_name == "k":
            retriever_args.setdefault("search_kwargs", {})[kwargs_name] = int(v[0])
    logger.info("retriever args: %s", retriever_args)
    return vectorstore.as_retriever(**retriever_args)


def _combine_documents(documents: list[Document]) -> list[dict]:
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


def _get_html_documents(urls: list[str]) -> Iterable[Document]:
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN)
    yield from splitter.split_documents(documents=html2text.transform_documents(docs))


def _get_pdf_documents(path: str) -> Iterable[Document]:
    loader = PyPDFLoader(path)
    yield from loader.load_and_split()


def _get_documents_from_input(input_schema: ParseResult) -> Iterable[Document]:
    if input_schema.scheme == "docs":
        yield from _get_file_documents(
            path=input_schema.netloc + input_schema.path,
            query=parse_qs(input_schema.query),
        )
    elif input_schema.scheme in ("http", "https"):
        yield from _get_html_documents(urls=[input_schema.geturl()])
    elif input_schema.scheme == "pdf":
        yield from _get_pdf_documents(path=input_schema.netloc + input_schema.path)


def _load_vectorstores(context: dict) -> list[FAISS]:
    if not context.get("input") or not context.get("embedding"):
        return []

    input_text = context["input"]

    vectorstores = []
    for raw_input in input_text.split("\n" * 3):
        if not raw_input:
            continue
        raw_input_stripped = raw_input.strip()
        if raw_input_stripped in (" ", "\n"):
            continue
        if raw_input_stripped.startswith("<<") and raw_input_stripped.endswith(">>"):
            input_schema = urlparse(raw_input_stripped[2:-2])
            input_schema_query = parse_qs(input_schema.query)
            if "embedding" in input_schema_query:
                embedding_schema = urlparse(input_schema_query["embedding"][0])
            else:
                embedding_schema = urlparse(context["embedding"])
            vectorstore = _get_vectorstore(
                embedding_schema=embedding_schema,
                input_schema=input_schema,
            )
            if vectorstore:
                vectorstores.append(vectorstore)

    return vectorstores


def _embed_image_from_text(input_text: str) -> Iterable[dict]:
    for raw_input in input_text.split("\n" * 3):
        if not raw_input:
            continue
        raw_input_stripped = raw_input.strip()
        if raw_input_stripped in (" ", "\n"):
            yield {"type": "text", "text": raw_input}
            continue
        if raw_input.strip().startswith("<<") and raw_input.strip().endswith(">>"):
            input_schema = urlparse(raw_input[2:-2])
            if input_schema.scheme == "image":
                data_url = image_to_data_url(input_schema.netloc + input_schema.path)
                yield {"type": "image_url", "image_url": {"url": data_url}}
                continue
        yield {"type": "text", "text": raw_input}
        continue


def _retrive_documents(context: dict) -> list[Document]:
    if (
        not context.get("vectorstores")
        or not context.get("embedding")
        or not context.get("input")
    ):
        return []

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
    retriever = _build_retriever(base_vectorstore, query)
    return retriever.invoke(input=context["input"])


def _combine_message(context: dict) -> list[BaseMessage]:
    messages = []
    if context.get("system"):
        messages.append(SystemMessage(content=context["system"]))
    if context.get("chat_history"):
        messages.extend(context["chat_history"])

    human_contents = []
    if context.get("input"):
        human_contents.extend(_embed_image_from_text(input_text=context["input"]))
    if context.get("documents"):
        human_contents.extend(_combine_documents(context["documents"]))
    messages.append(HumanMessage(content=human_contents))
    return messages


def build_chain(model: str) -> RunnableSerializable:
    model_schema = urlparse(model)
    chat_model = _detect_chat_model(model_schema=model_schema)

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
