# WikiArt Chatbot 👨‍🎨

An AI-powered chatbot that provides information about famous artworks using semantic search and LLM technology.

## Features

- 🤖 AI-powered art information assistant
- 🔍 Semantic search for finding relevant artworks
- 💬 Natural language conversation interface
- 🎯 Accurate and detailed art information
- 🚀 Fast and responsive UI

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Llama 3.2 model pulled in Ollama (`ollama pull llama3.2`)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/wikiart-chatbot.git
cd wikiart-chatbot
```

2. Create and activate a virtual environment:

```bash
python -m venv wikiart-env
source wikiart-env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package locally so it can be imported anywhere:

```bash
pip install -e .
```

## Usage

1. Make sure Ollama is running:

```bash
ollama serve
```

2. Run the chatbot:

```bash
python main.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:7860)

## Creating the FAISS index

The first time you run the chatbot it will automatically generate `wikiart_index.faiss` from `wikiart_metadata.csv`. If you prefer to create the index separately, run:

```bash
python build_index.py
```

## Project Structure

```
wikiart-chatbot/
├── wikiart_chatbot/
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Configuration settings
│   ├── exceptions.py    # Custom exceptions
│   ├── chatbot.py       # Main chatbot implementation
│   └── ui.py            # Gradio UI implementation
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── wikiart_metadata.csv
└── wikiart_index.faiss
```

## Configuration

You can customize the chatbot by modifying the default settings in `wikiart_chatbot/config.py`:

- Model selection
- Search parameters
- History length
- Timeout settings

## Running Tests

To run the unit tests, install the development dependencies and execute `pytest`:

```bash
pip install -e .[dev]
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for the LLM capabilities
- [Gradio](https://gradio.app/) for the UI framework
- [Sentence Transformers](https://www.sbert.net/) for semantic search
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
