# 📚 Document Chat Assistant (RAG Chatbot)

A small deployed AI app that lets you chat with your own documents. It
combines:

* **AI-1** — LLM API integration (Groq)
* **AI-2** — Engineered prompts (grounded, "say I don't know" behavior)
* **AI-3** — A working RAG (retrieval) pipeline using Chroma
* **AI-4** — A real chat web app (Streamlit), with loading \& error states

\---

## How it works (big picture)

```
Your documents (PDF/TXT/MD)
        │
        ▼
   ingest.py  ──► splits into chunks ──► embeds them ──► stores in Chroma
        │
        ▼
   (later, when a user asks a question)
        │
        ▼
     rag.py  ──► finds the most relevant chunks ──► builds a prompt
        │                                              │
        ▼                                              ▼
     app.py (chat UI)  ◄────────────────────  OpenAI API generates an answer
```

\---

## Step 1 — Get the project running locally

### 1a. Install Python dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

### 1b. Add your Groq API key (free, no card needed)

1. Get a free key from https://console.groq.com (API Keys section)
2. Copy `.env.example` to a new file named `.env`
3. Paste your key in:

```
   GROQ\\\_API\\\_KEY=gsk\_HVVOrHvLBfoTXxzSujLPWGdyb3FYdGzLQQJexzgVBIja1pAD23YC
   ```

**Never commit `.env` to git** — it's already listed in `.gitignore` so this
happens automatically.

### 1c. Add your documents

* A sample document is already in the `documents/` folder so you can test
right away.
* To use your own: drop `.pdf`, `.txt`, or `.md` files into `documents/`
(delete or keep the sample — your choice).

### 1d. Build the knowledge base

This reads your documents, chunks them, and creates the searchable vector
store:

```bash
python ingest.py
```

Re-run this any time you add or change documents.

### 1e. Run the chat app

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).
Try asking something like: *"How many vacation days do employees get?"*
(if you're using the sample document) — the answer should come straight
from the document text, not general knowledge.

### 1f. Verify retrieval is actually working (important!)

Ask a question the model could **not** know without your documents — e.g.
a made-up policy number, an internal term, or specific numbers from your
files. If it answers correctly, retrieval is working. If it gives a vague
generic answer, double check `ingest.py` ran successfully.

\---

## Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial RAG chatbot"
```

Create a new repo on GitHub, then:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

Because `.env` is in `.gitignore`, your API key will **not** be uploaded.
Double check by running `git status` — `.env` should not appear.

\---

## Step 3 — Deploy it live (free options)

### Option A: Streamlit Community Cloud (easiest)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app", pick your repo, branch `main`, and file `app.py`.
3. Under "Advanced settings" → "Secrets", add:

```
   GROQ\\\_API\\\_KEY = "gsk\_HVVOrHvLBfoTXxzSujLPWGdyb3FYdGzLQQJexzgVBIja1pAD23YC"
   ```

4. Click **Deploy**. You'll get a live URL like
`https://your-app-name.streamlit.app`.

> Note: Streamlit Cloud rebuilds the vector store fresh from your
> `documents/` folder since `chroma\\\_db/` isn't committed to git. If you
> want ingestion to happen automatically on deploy, see the note in
> "Troubleshooting" below.

### Option B: Render

1. Push your code to GitHub (Step 2 above).
2. Go to https://render.com → New → Web Service → connect your repo.
3. Set:

   * Build command: `pip install -r requirements.txt \\\&\\\& python ingest.py`
   * Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add an environment variable `GROQ\\\_API\\\_KEY` with your key.
5. Deploy. Render gives you a live URL.

\---

## Step 4 — Submit

Paste your live URL into the task's "Live Demo URL" field.

\---

## Project structure

```
rag-chatbot/
├── app.py                  # Streamlit chat UI
├── rag.py                  # Retrieval + prompt engineering + LLM call
├── ingest.py                # Builds the vector store from your documents
├── documents/                # Put your PDF/TXT/MD files here
│   └── sample\\\_company\\\_policy.txt
├── chroma\\\_db/                # Auto-created by ingest.py (don't edit manually)
├── requirements.txt
├── .env.example             # Template — copy to .env and fill in your key
├── .gitignore
└── README.md
```

\---

## Customizing it further

* **Change chunk size**: edit `CHUNK\\\_SIZE` / `CHUNK\\\_OVERLAP` in `ingest.py`.
* **Change how many chunks are retrieved**: edit `TOP\\\_K` in `rag.py`.
* **Change the model**: set `GROQ\\\_MODEL` in `.env` (see available models at
console.groq.com/docs/models — e.g. `llama-3.1-8b-instant` for faster,
lighter responses).
* **Switch to OpenAI/Anthropic instead of Groq**: replace the `call\\\_llm()`
function in `rag.py` with a call to that provider's API — the rest of the
pipeline (retrieval, prompting) stays exactly the same.
* **Switch vector stores**: Chroma is used here for simplicity; FAISS or
Pinecone would slot into the same spot in `rag.py`/`ingest.py`.

\---

## Troubleshooting

|Problem|Likely fix|
|-|-|
|`RuntimeError: No knowledge base found`|Run `python ingest.py` before starting the app|
|`GROQ\\\_API\\\_KEY is not set`|Check your `.env` file exists and has the right variable name|
|Answers seem generic / not from your docs|Re-run `ingest.py`, confirm your files are actually inside `documents/`|
|Deployment platform can't find `chroma\\\_db/`|Make sure your build command runs `python ingest.py`, since the vector store isn't committed to git (it's derived data)|



