# vector.py
import os
import time
import json
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("Missing PINECONE_API_KEY in .env")

# -----------------------------
# Pinecone initialization
# -----------------------------
pc = Pinecone(api_key=api_key)
index_name = "demo-index"
delete_existing_index = True  # Set to False to skip deletion during dev

# -----------------------------
# Delete index if exists
# -----------------------------
if delete_existing_index and index_name in pc.list_indexes().names():
    print(f"Deleting existing index: {index_name}")
    pc.delete_index(index_name)
    time.sleep(5)

# -----------------------------
# Create index
# -----------------------------
if index_name not in pc.list_indexes().names():
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# -----------------------------
# Connect to index
# -----------------------------
index = pc.Index(index_name)

# -----------------------------
# Load model and embed documents
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "Machine learning is a subfield of AI.",
    "Neural networks are inspired by the brain.",
    "Pizza is a popular Italian dish.",
    "Mount Everest is the highest mountain."
]
embeddings = model.encode(documents, normalize_embeddings=True).astype("float32")
print("Example vector:", embeddings[0][:5])

# -----------------------------
# Upload documents to Pinecone
# -----------------------------
vectors_to_upsert = [
    (f"doc-{i}", emb.tolist(), {"text": doc})
    for i, (doc, emb) in enumerate(zip(documents, embeddings))
]
index.upsert(vectors=vectors_to_upsert)
print(f"\n✅ Upserted {len(vectors_to_upsert)} vectors to '{index_name}'")

# -----------------------------
# Test multiple queries
# -----------------------------
test_queries = [
    "highest mountain",
    "tallest peak in the world",
    "Mount Everest",
    "famous mountain",
    "artificial intelligence",
    "French monument"
]


# -----------------------------
# Interactive query mode
# -----------------------------
while True:
    user_input = input("\n🔍 Enter a custom query (or type 'exit'): ").strip()
    if user_input.lower() == "exit":
        break

    query_vec = model.encode([user_input], normalize_embeddings=True).astype("float32")
    result = index.query(vector=query_vec[0].tolist(), top_k=1, include_metadata=True)

    print("\nTop 3 Results:")
    if result['matches']:
        for match in result['matches']:
            print(f"- {match['metadata']['text']} (score: {match['score']:.4f})")
    else:
        print("No matches found.")

print("\n✅ Script ran successfully")

