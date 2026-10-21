"""Central configuration for Visual Document Q&A.

Every setting is read from an environment variable (a ``.env`` file in the
project root is loaded automatically). See ``.env.example`` for the full list
with explanations.
"""

import os
from dataclasses import dataclass

from dotenv import dotenv_values

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env_file() -> None:
    """Copy values from the project's ``.env`` into ``os.environ``.

    Variables that are already set (and non-empty) in the real environment win.
    Safe to call repeatedly, so a key added to ``.env`` after start-up is found.
    """
    if not os.path.exists(ENV_FILE):
        return
    for name, value in dotenv_values(ENV_FILE).items():
        if value and not os.environ.get(name):
            os.environ[name] = value


load_env_file()

# Default models per LLM provider: (vision model, answer/chat model).
# Override them with VISION_MODEL / CHAT_MODEL if a provider renames its models.
DEFAULT_MODELS = {
    # "gemini-flash-latest" is Google's alias for the newest Flash model, which
    # is available on the free tier of the Gemini API.
    "gemini": ("gemini-flash-latest", "gemini-flash-latest"),
    "groq": (
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ),
    "ollama": ("llama3.2-vision", "llama3.2-vision"),
    "openai": ("gpt-4o-mini", "gpt-4o-mini"),
}

DEFAULT_EMBEDDING_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "gemini": "models/gemini-embedding-001",
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}

# Pause between vision calls (seconds) so free tiers are not rate limited.
# Gemini free tier allows roughly 10 requests/minute on Flash models.
DEFAULT_VISION_DELAY = {"gemini": 6.0, "groq": 2.0, "ollama": 0.0, "openai": 0.0}


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Build one with :meth:`Settings.from_env`."""

    llm_provider: str
    vision_model: str
    chat_model: str
    embedding_provider: str
    embedding_model: str
    chroma_dir: str
    image_dir: str
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int
    min_image_size: int
    max_images: int
    render_dpi: int
    vector_chart_min_drawings: int
    vision_delay_seconds: float
    max_retries: int
    vision_max_tokens: int
    answer_max_tokens: int
    ollama_keep_alive: str

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env("LLM_PROVIDER", "gemini").lower()
        if provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                f"Choose one of: {', '.join(DEFAULT_MODELS)}"
            )
        default_vision, default_chat = DEFAULT_MODELS[provider]

        embedding_provider = _env("EMBEDDING_PROVIDER", "fastembed").lower()
        if embedding_provider not in DEFAULT_EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER '{embedding_provider}'. "
                f"Choose one of: {', '.join(DEFAULT_EMBEDDING_MODELS)}"
            )

        return cls(
            llm_provider=provider,
            vision_model=_env("VISION_MODEL", default_vision),
            chat_model=_env("CHAT_MODEL", default_chat),
            embedding_provider=embedding_provider,
            embedding_model=_env(
                "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS[embedding_provider]
            ),
            chroma_dir=_env("CHROMA_DIR", "./chroma_data"),
            image_dir=_env("IMAGE_DIR", "./temp_images"),
            retrieval_k=_env_int("RETRIEVAL_K", 6),
            chunk_size=_env_int("CHUNK_SIZE", 1000),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
            min_image_size=_env_int("MIN_IMAGE_SIZE", 100),
            max_images=_env_int("MAX_IMAGES", 30),
            render_dpi=_env_int("RENDER_DPI", 150),
            vector_chart_min_drawings=_env_int("VECTOR_CHART_MIN_DRAWINGS", 40),
            vision_delay_seconds=_env_float(
                "VISION_DELAY_SECONDS", DEFAULT_VISION_DELAY[provider]
            ),
            max_retries=_env_int("MAX_RETRIES", 6),
            # Lower these for local CPU models: generation speed is the bottleneck.
            vision_max_tokens=_env_int("VISION_MAX_TOKENS", 1500),
            answer_max_tokens=_env_int("ANSWER_MAX_TOKENS", 1500),
            # How long Ollama keeps the model in memory between requests.
            ollama_keep_alive=_env("OLLAMA_KEEP_ALIVE", "30m"),
        )


settings = Settings.from_env()
