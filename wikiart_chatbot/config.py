"""Configuration settings for the WikiArt Chatbot."""

from dataclasses import dataclass

# Constants
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TOP_K = 3
DEFAULT_MAX_HISTORY = 10
DEFAULT_TIMEOUT = 30

@dataclass
class Config:
    """Configuration for the WikiArt Chatbot."""
    model: str = DEFAULT_MODEL
    top_k: int = DEFAULT_TOP_K
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    max_history: int = DEFAULT_MAX_HISTORY
    timeout: int = DEFAULT_TIMEOUT 