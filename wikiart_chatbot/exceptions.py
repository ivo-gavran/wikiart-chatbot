"""Custom exceptions for the WikiArt Chatbot."""

class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass

class SearchError(Exception):
    """Base exception for search-related errors."""
    pass 