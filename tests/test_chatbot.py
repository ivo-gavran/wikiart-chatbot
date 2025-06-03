import pandas as pd
from wikiart_chatbot.chatbot import WikiArtChatbot


def test_format_artwork_info():
    chatbot = object.__new__(WikiArtChatbot)
    row = pd.Series({
        "title": "Mona Lisa",
        "artist": "Leonardo da Vinci",
        "year": 1503,
        "style": "Renaissance",
        "genre": "Portrait",
        "description": "A painting of Mona Lisa"
    })
    expected = (
        "Title: Mona Lisa\n"
        "Artist: Leonardo da Vinci\n"
        "Year: 1503\n"
        "Style: Renaissance\n"
        "Genre: Portrait\n"
        "Description: A painting of Mona Lisa"
    )
    assert chatbot.format_artwork_info(row) == expected
