"""Utility script to create the FAISS index for WikiArt metadata."""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from wikiart_chatbot.config import Config


def build_index():
    """Generate and save the FAISS index from the metadata."""
    config = Config()
    metadata_path = Path("wikiart_metadata.csv")
    index_path = Path(config.index_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path, quoting=1)
    texts = df.apply(
        lambda row: f"{row['title']} by {row['artist']} - {row['style']} ({row['year']}): {row['description']}",
        axis=1,
    )

    embedding_model = SentenceTransformer(config.embedding_model)
    embeddings = embedding_model.encode(texts.tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, str(index_path))
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    build_index()
