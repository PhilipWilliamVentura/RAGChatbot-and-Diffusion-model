from fastapi import FastAPI
from pydantic import BaseModel
from rag import query_rag
from ex_diffusion import generate_diagram
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://quantmllabs.vercel.app"],  # your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    text: str
    question: str
    diagram: str

@app.post("/ask")
def ask(req: AskRequest):
    # Step 1: RAG for text answer
    answer = query_rag(req.question, req.text)

    # Step 2: Generate diagram
    diagram_path = generate_diagram(f"diagram explaining: {req.diagram}")

    return {
        "answer": answer,
        "image_url": f"http://localhost:8000/{diagram_path}"
    }

# Serve generated images
app.mount("/generated", StaticFiles(directory="generated"), name="generated")
