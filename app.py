"""
app.py
------
The chat UI (AI-4: wrapping everything in a real web app).

Run locally with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag import answer_question

load_dotenv()  # reads GROQ_API_KEY from your .env file when running locally

st.set_page_config(page_title="Document Chat Assistant", page_icon="📚")

# --- Auto-build the knowledge base on first launch (needed for Streamlit Cloud, ---
# --- which doesn't run a separate build command like "python ingest.py") ---
if not os.path.exists("chroma_db"):
    with st.spinner("Setting up knowledge base for the first time... (~30 seconds)"):
        from ingest import build_vector_store
        build_vector_store()

st.title("📚 Document Chat Assistant")
st.caption(
    "Ask me anything about the documents I was given. "
    "I'll only answer using what's actually in them."
)

# Keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate and show the assistant's response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")  # loading state

        try:
            result = answer_question(user_question)
            answer_text = result["answer"]
            sources = result["sources"]

            if sources:
                answer_text += "\n\n---\n📄 **Sources:** " + ", ".join(sources)

            placeholder.markdown(answer_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text}
            )

        except RuntimeError as e:
            # Expected, "friendly" errors (missing API key, no knowledge base yet)
            error_msg = f"⚠️ {str(e)}"
            placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

        except Exception as e:
            # Anything unexpected (network issue, API downtime, etc.)
            error_msg = (
                "❌ Something went wrong while generating a response. "
                f"Details: {str(e)}"
            )
            placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# Sidebar with quick instructions
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This assistant retrieves relevant chunks from your documents "
        "before answering, so its responses are grounded in your actual "
        "source material rather than general knowledge."
    )
    st.write("Powered by Groq's free API (Llama 3.3).")
    st.write("**To add documents:**")
    st.code("1. Drop files into /documents\n2. Run: python ingest.py\n3. Restart the app")
