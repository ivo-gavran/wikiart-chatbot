"""Main chatbot implementation."""

import pandas as pd
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from .config import Config
from .exceptions import OllamaError, SearchError

# Configure logging
logger = logging.getLogger(__name__)

class WikiArtChatbot:
    """A chatbot that provides information about artworks using semantic search and LLM."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the chatbot with optional configuration.
        
        Args:
            config: Optional configuration object. If not provided, uses default values.
        """
        self.config = config or Config()
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        self.load_data()
        self.conversation_history: List[Dict] = []

    def load_data(self) -> None:
        """Load and prepare the data for the chatbot.
        
        Raises:
            FileNotFoundError: If the metadata file is not found.
        """
        metadata_path = Path("wikiart_metadata.csv")
        index_path = Path("wikiart_index.faiss")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path, quoting=1)  # QUOTE_ALL mode to properly handle commas in fields
        
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            logger.info("Creating FAISS index...")
            texts = self.df.apply(
                lambda row: f"{row['title']} by {row['artist']} - {row['style']} ({row['year']}): {row['description']}", 
                axis=1
            )
            embeddings = self.embedding_model.encode(texts.tolist(), show_progress_bar=True)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))
            faiss.write_index(self.index, str(index_path))
            logger.info("Saved FAISS index.")

    def search_wikiart(self, query: str, top_k: Optional[int] = None) -> pd.DataFrame:
        """Perform semantic search on the artwork database.
        
        Args:
            query: The search query string.
            top_k: Optional number of results to return. Defaults to config value.
            
        Returns:
            DataFrame containing the search results.
            
        Raises:
            SearchError: If the search operation fails.
        """
        try:
            top_k = top_k or self.config.top_k
            embedding = self.embedding_model.encode([query])
            _, indices = self.index.search(np.array(embedding), top_k)
            return self.df.iloc[indices[0]]
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise SearchError(f"Failed to perform search: {str(e)}")

    def query_ollama(self, context: str, user_input: str) -> str:
        """Query the Ollama model with improved prompt engineering.
        
        Args:
            context: The context information about artworks.
            user_input: The user's question or input.
            
        Returns:
            The model's response as a string.
            
        Raises:
            OllamaError: If the query to Ollama fails.
        """
        try:
            prompt = self._create_prompt(context, user_input)
            response = self._make_ollama_request(prompt)
            return self._process_ollama_response(response)
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama API")
            raise OllamaError("Could not connect to the AI model. Please ensure Ollama is running.")
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama API timed out")
            raise OllamaError("The request to the AI model timed out. Please try again.")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama query: {str(e)}")
            raise OllamaError(f"An unexpected error occurred: {str(e)}")

    def _create_prompt(self, context: str, user_input: str) -> str:
        """Create the prompt for the Ollama model.
        
        Args:
            context: The context information about artworks.
            user_input: The user's question or input.
            
        Returns:
            The formatted prompt string.
        """
        return f"""You are an art expert assistant. Use the following context to provide a detailed and engaging answer to the user's question.
        Be specific about the artworks mentioned and their historical significance.

        Context:
        {context}

        Question:
        {user_input}

        Provide a well-structured answer that:
        1. Directly addresses the user's question
        2. References specific artworks when relevant
        3. Includes interesting historical or artistic details
        4. Maintains a conversational and engaging tone

        Answer:"""

    def _make_ollama_request(self, prompt: str) -> requests.Response:
        """Make the request to the Ollama API.
        
        Args:
            prompt: The prompt to send to Ollama.
            
        Returns:
            The response from the Ollama API.
            
        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        return requests.post(
            self.config.ollama_url,
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=self.config.timeout
        )

    def _process_ollama_response(self, response: requests.Response) -> str:
        """Process the response from the Ollama API.
        
        Args:
            response: The response from the Ollama API.
            
        Returns:
            The processed response text.
            
        Raises:
            OllamaError: If the response cannot be processed.
        """
        if response.status_code != 200:
            logger.error(f"Ollama API error: Status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            raise OllamaError(f"Received error from AI model (Status code: {response.status_code})")
        
        response_data = response.json()
        if "response" not in response_data:
            logger.error(f"Unexpected response format: {response_data}")
            raise OllamaError("Received unexpected response format from AI model")
            
        return response_data["response"]

    def format_artwork_info(self, row: pd.Series) -> str:
        """Format artwork information in a structured way.
        
        Args:
            row: A pandas Series containing artwork information.
            
        Returns:
            Formatted string containing the artwork information.
        """
        return f"""Title: {row['title']}
Artist: {row['artist']}
Year: {row['year']}
Style: {row['style']}
Genre: {row['genre']}
Description: {row['description']}"""

    def process_message(self, message: str, history: Optional[List[Dict]]) -> Tuple[str, List[Dict]]:
        """Process a user message and return the response with updated history.
        
        Args:
            message: The user's message.
            history: The current conversation history. If ``None``, a new
                history list will be created.
            
        Returns:
            Tuple containing an empty string and the updated history.
        """
        history = history or []
        try:
            matches = self.search_wikiart(message)
            if matches.empty:
                response = "I couldn't find any relevant artworks to answer your question."
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history

            context = "\n\n".join(self.format_artwork_info(row) for _, row in matches.iterrows())
            response = self.query_ollama(context, message)
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            if len(history) > self.config.max_history * 2:
                history = history[-(self.config.max_history * 2):]
            
            return "", history
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_msg = f"I encountered an error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history 