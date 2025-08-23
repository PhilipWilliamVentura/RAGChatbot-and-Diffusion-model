# Quant ML Labs Backend

This repository contains the **backend for Quant ML Labs**, powering the AI companion, RAG, and Diffusion functionalities for the [frontend app](https://quantmllabs.vercel.app).

The backend is fully built in **Python** using **FastAPI**, **PyTorch**, and **LangChain**, and includes:  

- A **Diffusion Model built from scratch**, including:
  - Variational Autoencoder (VAE)
  - CLIP embeddings
  - Self-Attention & Cross-Attention
  - UNet architecture
  - DDPM sampling
  - Pipeline connecting all components for image generation
- **RAG (Retrieval-Augmented Generation)** implementation with **LangChain** and **FAISS**
- **FastAPI server** exposing endpoints for real-time interaction with the frontend app

---

## 🔥 Features

- **Diffusion Model from Scratch:** Generate images/diagrams from textual prompts or images.
- **RAG Question-Answering:** Retrieve answers from user-provided text using embeddings and FAISS.
- **Full-Stack Integration:** Backend powers the interactive AI companion on [Quant ML Labs](https://quantmllabs.vercel.app).

---

## 🚀 Installation & Setup

Clone the Repository

```bash
git clone https://github.com/PhilipWilliamVentura/RAGChatbot-and-Diffusion-model.git
cd RAGChatbot-and-Diffusion-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Prepare the Data Folder for the Diffusion Model
Create the following folder structure:

```bash
sd/data/
├─ merges.txt
├─ vocab.json
├─ v1-5-pruned-emaonly.ckpt
```

Download the required files:

Tokenizer files:  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer

vocab.json

merges.txt

Model weights:  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main

v1-5-pruned-emaonly.ckpt

Save all files in sd/data/.

Running the Backend
Start the FastAPI server:

```bash
uvicorn main:app --reload --port 8000
```
Endpoint: /ask — Accepts JSON with text and question, returns an answer and optional generated diagram.

Ensure the backend is running before using the frontend app.
