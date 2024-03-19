import logging
from hashlib import sha256
from operator import itemgetter
from pathlib import Path
from urllib.parse import ParseResult, parse_qs, urlparse

import boto3
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain_community.chat_models import BedrockChat, ChatOllama, ChatOpenAI
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import (
    BedrockEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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

DOCUMENT_PROMPT = PromptTemplate.from_template(
    """#source:{source}
```{language}
{page_content}
```""",
)


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


def _get_faiss_vectorstore(
    documents: list[Document],
    embedding: Embeddings,
    folder_dir: Path,
) -> FAISS:
    if folder_dir.exists():
        return FAISS.load_local(
            folder_path=str(folder_dir),
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )
    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)
    vectorstore.save_local(folder_path=str(folder_dir))
    return vectorstore


def _get_documents(document_schema: ParseResult) -> VectorStoreRetriever:
    document_args = {}
    query = parse_qs(document_schema.query)
    if query.get("glob"):
        document_args["glob"] = query["glob"][0]
    if query.get("exclude"):
        document_args["exclude"] = query["exclude"]
    if query.get("show_progress"):
        document_args["show_progress"] = bool(query["show_progress"][0])
    if query.get("suffixes"):
        document_args["suffixes"] = query["suffixes"]

    if query.get("faiss_folder"):
        folder_dir = Path(query["faiss_folder"][0])
    else:
        folder_dir = Path.joinpath(
            Path.home(),
            ".cache",
            "faiss",
            sha256(document_schema.geturl().encode()).hexdigest(),
        )

    logger.info("documents args: %s", document_args)
    loader = GenericLoader.from_filesystem(
        path=document_schema.path,
        parser=LanguageParser(),
        **document_args,
    )
    documents = loader.load()
    if query.get("language"):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language(query["language"][0]),
        )
        documents = splitter.split_documents(documents=documents)

    embedding = _detect_embedding(embedding_schema=document_schema)

    vectorstore = _get_faiss_vectorstore(
        documents=documents,
        embedding=embedding,
        folder_dir=folder_dir,
    )

    retriever_args = {}
    if query.get("search_type"):
        retriever_args["search_type"] = query["search_type"][0]
    for k, v in [(k, v) for k, v in query.items() if k.startswith("search_kwargs_")]:
        retriever_args.setdefault("search_kwargs", {})
        retriever_args.setdefault("search_kwargs", {})[k[15:]] = v[0]
    logger.info("retriever args: %s", retriever_args)
    return vectorstore.as_retriever(**retriever_args)


def _combine_documents(documents: list[Document]) -> list[BaseMessage]:
    return [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": format_document(doc, DOCUMENT_PROMPT),
                }
                for doc in documents
            ],
        ),
    ]


def _combine_message(context: dict) -> list[BaseMessage]:
    messages = []
    human_message_contents = []
    if context.get("system"):
        messages.append(SystemMessage(content=context["system"]))
    if context.get("documents"):
        messages.extend(_combine_documents(context["documents"]))
    if context.get("chat_history"):
        messages.extend(context["chat_history"])
    if context.get("input"):
        for raw_input in context["input"].split("\n\n"):
            if raw_input.startswith("<<file://") and raw_input.endswith(">>"):
                file_url = urlparse(raw_input[2:-2])
                data_url = image_to_data_url(f"{file_url.netloc}{file_url.path}")
                human_message_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                )
            else:
                human_message_contents.append(
                    {
                        "type": "text",
                        "text": raw_input,
                    },
                )
    if human_message_contents:
        messages.append(HumanMessage(content=human_message_contents))
    return messages


def build_chain(
    model: str,
    documents: str,
) -> RunnableSerializable:
    model_schema = urlparse(model)
    chat_model = _detect_chat_model(model_schema=model_schema)

    if documents:
        document_schema = urlparse(documents)
        retriever = _get_documents(document_schema=document_schema)
        documents_loader = RunnablePassthrough.assign(
            documents=itemgetter("input") | retriever,
        )
    else:
        documents_loader = RunnablePassthrough()

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
        | documents_loader
        | RunnablePassthrough.assign(messages=RunnableLambda(_combine_message))
        | RunnablePassthrough.assign(
            answer=itemgetter("messages") | chat_model | StrOutputParser(),
        )
        | memory_saver
    )
