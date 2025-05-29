import pandas as pd
import numpy as np
import faiss
import os
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer

# File paths
METADATA_FILE = "wikiart_metadata.csv"
FAISS_INDEX_FILE = "wikiart_index.faiss"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load CSV
if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(f"{METADATA_FILE} not found!")

df = pd.read_csv(METADATA_FILE)

# Build or load FAISS index
if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    print("Creating FAISS index...")
    texts = df.apply(lambda row: f"{row['title']} by {row['artist']} - {row['style']} ({row['year']}): {row['description']}", axis=1)
    embeddings = embedding_model.encode(texts.tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("Saved FAISS index.")

# Semantic search
def search_wikiart(query, top_k=3):
    embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(embedding), top_k)
    return df.iloc[indices[0]]

# Query Ollama
def query_ollama(context, user_input, model="llama3.2:latest"):
    prompt = f"""You are an art expert. Use the following context to answer the user's question.

Context:
{context}

Question:
{user_input}

Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "No response from Ollama.")

# Chatbot logic
def chatbot(user_input):
    matches = search_wikiart(user_input)
    context = "\n\n".join(
        f"Title: {row['title']}\nArtist: {row['artist']}\nYear: {row['year']}\nStyle: {row['style']}\nGenre: {row['genre']}\nDescription: {row['description']}"
        for _, row in matches.iterrows()
    )
    return query_ollama(context, user_input)

# Launch UI
gr.Interface(fn=chatbot, inputs="text", outputs="text", title="WikiArt Chatbot (Local + Ollama)").launch()
