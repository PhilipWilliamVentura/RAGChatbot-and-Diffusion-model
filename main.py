from fastapi import FastAPI
from pydantic import BaseModel
from rag import query_rag
from diffusion import generate_diagram
from fastapi.staticfiles import StaticFiles

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    # Step 1: RAG for text answer
    answer = query_rag(req.question)

    # Step 2: Generate diagram
    diagram_path = generate_diagram(f"diagram explaining: {answer}")

    return {
        "answer": answer,
        "image_url": f"http://localhost:5000/{diagram_path}"
    }

app.mount("/generated", StaticFiles(directory="generated"), name="generated")