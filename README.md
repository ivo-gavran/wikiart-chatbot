# WikiArt Chatbot

A local chatbot that provides information about artwork using semantic search and the Ollama LLM. The chatbot uses WikiArt metadata to answer questions about paintings, artists, styles, and art history.

## Features

- Semantic search using FAISS and Sentence Transformers
- Local LLM integration with Ollama
- Interactive Gradio web interface
- Artwork information including titles, artists, styles, and descriptions
- Fast and efficient vector similarity search

## Prerequisites

- Python 3.7+
- Ollama installed and running locally
- Required Python packages (see Installation section)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/wikiart-chatbot.git
cd wikiart-chatbot
```

2. Create and activate a virtual environment:

```bash
python -m venv wikiart-env
source wikiart-env/bin/activate
```

3. Install required packages:

```bash
pip install pandas numpy faiss-cpu requests gradio sentence-transformers
```

4. Make sure Ollama is installed and running locally (default port: 11434)

## Usage

1. Start the chatbot:

```bash
python wikiart_chatbot.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

3. Type your questions about art in the text input field and press Enter to get responses

## How It Works

1. The chatbot uses a pre-trained sentence transformer model (all-MiniLM-L6-v2) to create embeddings of artwork descriptions
2. When you ask a question, it performs semantic search to find relevant artwork information
3. The context from the search results is sent to Ollama (using llama3.2 model) to generate a response
4. The response is displayed in the web interface

## Project Structure

- `wikiart_chatbot.py`: Main application file
- `wikiart_metadata.csv`: Dataset containing artwork information
- `wikiart_index.faiss`: FAISS index file for fast similarity search

## Requirements

- pandas
- numpy
- faiss-cpu
- requests
- gradio
- sentence-transformers
- Ollama (running locally)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
