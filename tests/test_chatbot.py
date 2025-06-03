import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wikiart_chatbot.chatbot import WikiArtChatbot
from wikiart_chatbot.config import Config
from wikiart_chatbot.exceptions import OllamaError


def create_bot():
    bot = WikiArtChatbot.__new__(WikiArtChatbot)
    bot.config = Config()
    return bot


def test_make_ollama_request_raises_for_status():
    bot = create_bot()
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    with patch('requests.post', return_value=mock_response) as mock_post:
        with pytest.raises(requests.exceptions.HTTPError):
            bot._make_ollama_request('prompt')
        mock_post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()


def test_process_ollama_response_success():
    bot = create_bot()
    mock_response = MagicMock()
    mock_response.json.return_value = {'response': 'hi'}
    assert bot._process_ollama_response(mock_response) == 'hi'


def test_process_ollama_response_missing_key():
    bot = create_bot()
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    with pytest.raises(OllamaError):
        bot._process_ollama_response(mock_response)
