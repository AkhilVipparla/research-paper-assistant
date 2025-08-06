import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from PyPDF2 import PdfReader

genai.configure(api_key="AIzaSyCh2lIDbk-2VE-dz0f0IsWk8mzBQP5jpnw")
llm_model = genai.GenerativeModel("gemini-1.5-pro")
embedding_model = "models/embedding-001"


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    return [np.array(genai.embed_content(model=embedding_model, content=chunk, task_type="retrieval_document")["embedding"], dtype='float32') for chunk in chunks]

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(embeddings))
    return index

def semantic_retrieve(query, index, chunks, k=4):
    query_embed = np.array([genai.embed_content(model=embedding_model, content=query, task_type="retrieval_query")["embedding"]], dtype='float32')
    distances, indices = index.search(query_embed, k)
    return [chunks[i] for i in indices[0]]


def decide_task_type(query):
    prompt = f"""
You're an intelligent assistant that chooses the right task type based on a user's query about a research paper.

Choose one of the following tools:
- "qa" for answering specific questions from the paper.
- "summarize" if the user asks to summarize content.
- "generate" for generating new content (e.g. abstract, title, conclusion).

User query: "{query}"

Respond with only one word: qa, summarize, or generate.
"""
    decision = llm_model.generate_content(prompt).text.strip().lower()
    return decision if decision in ["qa", "summarize", "generate"] else "qa"


def execute_agent_task(query, task_type, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    
    if task_type == "qa":
        prompt = f"""You are an assistant answering questions based on context.

Context:
{context}

Question: {query}

Answer using only the above context.
If not enough information is available, say "The answer is not in the document."
"""
    elif task_type == "summarize":
        prompt = f"""Summarize the following context from a research paper:

{context}

Be concise and preserve key technical points.
"""
    elif task_type == "generate":
        prompt = f"""Generate new content for a research paper based on the following context.

Context:
{context}

User's instruction: {query}
"""
    else:
        return "Unknown task."

    response = llm_model.generate_content(prompt)
    return response.text.strip()

st.set_page_config(page_title="Agentic RAG Research Paper Assistant", layout="wide")
st.title("Research Paper Assistant")

uploaded_file = st.file_uploader("Upload a research paper PDF", type=["pdf"])
query = st.text_input("Ask a question!!")

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.chunks = []

if uploaded_file and st.button("Process Document"):
    with st.spinner("Reading and embedding..."):
        full_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(full_text)
        embeddings = embed_chunks(chunks)
        faiss_index = build_faiss_index(embeddings)

        st.session_state.faiss_index = faiss_index
        st.session_state.chunks = chunks

    st.success("Document processed!")

if query and st.session_state.faiss_index:
    with st.spinner(""):
        task = decide_task_type(query)
        context = semantic_retrieve(query, st.session_state.faiss_index, st.session_state.chunks)
        answer = execute_agent_task(query, task, context)
        st.markdown(f"**tool selected:** `{task}`")
        st.markdown("### Response")
        st.write(answer)
