"""
rag.py
------
The "brain" of the app. Given a user's question, this file:
1. Retrieves the most relevant chunks from the Chroma vector store (AI-3).
2. Builds a carefully engineered prompt using those chunks (AI-2).
3. Calls the LLM API to generate a grounded answer (AI-1).

app.py imports and calls `answer_question()` from this file.
"""

import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "my_knowledge_base"
TOP_K = 4  # how many chunks to retrieve per question

# --- Set up the embedding function (must match the one used in ingest.py) ---
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# --- Connect to the persisted vector store ---
_client = chromadb.PersistentClient(path=CHROMA_PATH)


def _get_collection():
    """Fetch the collection, raising a clear error if ingest.py hasn't run yet."""
    try:
        return _client.get_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )
    except Exception:
        raise RuntimeError(
            "No knowledge base found. Run 'python ingest.py' first "
            "to build the vector store from your documents."
        )


def retrieve_chunks(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k most relevant chunks for the query, with their source."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(documents, metadatas):
        chunks.append({"text": doc, "source": meta.get("source", "unknown")})

    return chunks


# --- This is the engineered system prompt (AI-2: prompt engineering) ---
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY \
the context provided below, which comes from the user's own documents.

Rules you must follow:
1. Base your answer strictly on the given context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say clearly: \
"I don't have enough information in the provided documents to answer that."
3. When possible, mention which source document your answer came from.
4. Be concise and direct. Do not pad your answer with filler.
5. Never make up facts, sources, or details that are not in the context.
"""


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    if not chunks:
        context_block = "(No relevant context was found in the documents.)"
    else:
        context_block = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )

    return f"""Context from documents:
---
{context_block}
---

Question: {query}

Answer the question using only the context above."""


def call_llm(system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "(see .env.example). Get a free key at console.groq.com"
        )

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    # Groq's API is OpenAI-compatible, so we just point the OpenAI client
    # at Groq's base URL instead of OpenAI's.
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature = more grounded, less "creative"
    )

    return response.choices[0].message.content


def answer_question(query: str) -> dict:
    """
    Full RAG pipeline: retrieve -> prompt -> generate.
    Returns a dict with the answer text and the sources used,
    so the UI can display both.
    """
    chunks = retrieve_chunks(query)
    user_prompt = build_user_prompt(query, chunks)
    answer = call_llm(SYSTEM_PROMPT, user_prompt)

    sources = sorted({c["source"] for c in chunks})
    return {"answer": answer, "sources": sources}
