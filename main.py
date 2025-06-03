"""Main entry point for the WikiArt Chatbot."""

import logging
from wikiart_chatbot.ui import create_ui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the WikiArt Chatbot."""
    interface = create_ui()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
