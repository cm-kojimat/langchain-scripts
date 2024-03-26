import logging
from hashlib import sha256
from pathlib import Path
from urllib.parse import ParseResult, parse_qs

from langchain.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_scripts.core.documents_loader import get_documents_from_input
from langchain_scripts.core.embeddings import detect_embedding

logger = logging.getLogger(__name__)


def _is_existing_faiss_vectorstore(
    folder_dir: Path,
    index_name: str = "index",
) -> bool:
    return Path(folder_dir).joinpath(f"{index_name}.faiss").exists()


def get_vectorstore(
    embedding_schema: ParseResult,
    input_schema: ParseResult,
) -> FAISS | None:
    query = parse_qs(embedding_schema.query)
    embedding = detect_embedding(embedding_schema=embedding_schema)

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

    documents = list(get_documents_from_input(input_schema=input_schema))
    if not documents:
        return None

    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)
    vectorstore.save_local(folder_path=str(folder_dir), index_name=index_name)
    return vectorstore


def build_retriever(
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
