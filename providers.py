"""Factories for the chat, vision and embedding models.

The rest of the app only talks to the LangChain interfaces returned here
(``BaseChatModel`` and ``Embeddings``), so switching between Gemini, Groq,
Ollama or OpenAI is purely a configuration change (see ``config.py``).

Provider packages are imported lazily, so only the one you use must be
installed and only its API key must be set.
"""

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import ENV_FILE, load_env_file, settings

API_KEY_ENV = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _require_key(provider: str) -> None:
    env_name = API_KEY_ENV.get(provider)
    if not env_name or os.getenv(env_name):
        return
    load_env_file()  # the key may have been added to .env after start-up
    if not os.getenv(env_name):
        raise RuntimeError(
            f"{env_name} is not set. Add it to {ENV_FILE} "
            f"(needed because the '{provider}' provider is selected)."
        )


def _build_chat_model(provider: str, model: str, max_tokens: int) -> BaseChatModel:
    _require_key(provider)
    common = {"temperature": 0, "max_retries": settings.max_retries}

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, max_output_tokens=max_tokens, **common)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, max_tokens=max_tokens, **common)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Ollama runs locally, so retries are not needed. keep_alive avoids
        # reloading the model (slow on CPU) between requests.
        return ChatOllama(
            model=model,
            temperature=0,
            num_predict=max_tokens,
            keep_alive=settings.ollama_keep_alive,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, max_tokens=max_tokens, **common)
    raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=1)
def get_vision_model() -> BaseChatModel:
    """Model used to describe images (charts, tables, diagrams, scanned pages)."""
    return _build_chat_model(
        settings.llm_provider, settings.vision_model, max_tokens=settings.vision_max_tokens
    )


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Model used to write the final answer from the retrieved context."""
    return _build_chat_model(
        settings.llm_provider, settings.chat_model, max_tokens=settings.answer_max_tokens
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Embedding model used to index and search the document."""
    provider, model = settings.embedding_provider, settings.embedding_model

    if provider == "fastembed":
        # Runs locally on CPU: free, private and has no rate limits.
        # The model (~70 MB) is downloaded once on first use.
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=model)
    if provider == "gemini":
        _require_key(provider)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model)
    if provider == "openai":
        _require_key(provider)
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    raise ValueError(f"Unsupported embedding provider: {provider}")
