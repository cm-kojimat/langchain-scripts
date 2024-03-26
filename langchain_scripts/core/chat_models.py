from functools import lru_cache
from urllib.parse import ParseResult, parse_qs

import boto3
from langchain_community.chat_models import BedrockChat, ChatOllama, ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


@lru_cache(maxsize=8)
def detect_chat_model(model_schema: ParseResult) -> BaseChatModel:
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
