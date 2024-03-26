import logging
from urllib.parse import ParseResult, parse_qs

from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _get_file_documents(path: str, query: dict) -> list[Document]:
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
    return documents


def _get_html_documents(urls: list[str]) -> list[Document]:
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN)
    return splitter.split_documents(documents=html2text.transform_documents(docs))


def _get_pdf_documents(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    return loader.load_and_split()


def get_documents_from_input(input_schema: ParseResult) -> list[Document]:
    if input_schema.scheme == "docs":
        return _get_file_documents(
            path=input_schema.netloc + input_schema.path,
            query=parse_qs(input_schema.query),
        )

    if input_schema.scheme in ("http", "https"):
        return _get_html_documents(urls=[input_schema.geturl()])

    if input_schema.scheme == "pdf":
        return _get_pdf_documents(path=input_schema.netloc + input_schema.path)

    return []
