"""
ingest.py
---------
Run this ONCE (and again any time you add/change documents) to build
your searchable knowledge base.

What it does:
1. Reads every .txt, .md, and .pdf file inside the "documents" folder.
2. Splits each document into small overlapping chunks.
3. Converts each chunk into a vector embedding (using a free, local
   embedding model, so this step needs no API key).
4. Saves everything into a local Chroma vector database on disk
   (in the "chroma_db" folder) so app.py can search it later.

Usage:
    python ingest.py
"""

import os
import glob
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

DOCUMENTS_FOLDER = "documents"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "my_knowledge_base"

# Chunking settings: how big each chunk is, and how much chunks overlap
# (overlap helps avoid cutting an important sentence in half between chunks)
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks


def read_txt_or_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def load_documents() -> list[dict]:
    """Return a list of {"source": filename, "text": full_text} dicts."""
    docs = []
    patterns = ["*.txt", "*.md", "*.pdf"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, pattern)))

    if not files:
        print(f"⚠️  No documents found in '{DOCUMENTS_FOLDER}/'. "
              f"Add some .txt, .md, or .pdf files there first.")
        return docs

    for path in files:
        filename = os.path.basename(path)
        try:
            if path.lower().endswith(".pdf"):
                text = read_pdf(path)
            else:
                text = read_txt_or_md(path)

            if text.strip():
                docs.append({"source": filename, "text": text})
                print(f"✅ Loaded {filename} ({len(text)} characters)")
            else:
                print(f"⚠️  {filename} appears to be empty, skipping.")
        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window chunker."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide forward, keeping overlap

    return chunks


def build_vector_store():
    docs = load_documents()
    if not docs:
        return

    print("\n📦 Setting up embedding model (lightweight, ONNX-based)...")
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Start fresh each time you re-run ingest, so stale chunks don't linger
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['source']}-{i}")
            all_metadatas.append({"source": doc["source"]})

    print(f"\n🔍 Embedding {len(all_chunks)} chunks from {len(docs)} document(s)...")

    # Add in batches so it works even for larger document sets
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i + batch_size],
            ids=all_ids[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size],
        )

    print(f"\n✅ Done! Vector store saved to '{CHROMA_PATH}/'.")
    print(f"   {len(all_chunks)} chunks are now searchable.")
    print("\nNext step: run 'streamlit run app.py' to chat with your documents.")


if __name__ == "__main__":
    build_vector_store()
