# rag.py
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def build_faiss_index(chunks: List[str]):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve_chunks(question: str, chunks: List[str], index, embeddings, top_k=3):
    q_vec = embedder.encode([question], convert_to_tensor=False)
    q_vec = np.array(q_vec).astype("float32")
    D, I = index.search(q_vec, top_k)
    return [chunks[i] for i in I[0]]


def query_rag(text: str, question: str) -> str:
    chunks = text_splitter.split_text(text)

    index, embeddings = build_faiss_index(chunks)

    retrieved_chunks = retrieve_chunks(question, chunks, index, embeddings, top_k=3)

    context = "\n".join(retrieved_chunks)
    prompt = f"""
You are an assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text
