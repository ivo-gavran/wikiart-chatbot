"""WikiArt Chatbot - An AI-powered art information assistant."""

__version__ = "1.0.0"

from .chatbot import WikiArtChatbot
from .config import Config
from .exceptions import OllamaError, SearchError

__all__ = ["WikiArtChatbot", "Config", "OllamaError", "SearchError"]
