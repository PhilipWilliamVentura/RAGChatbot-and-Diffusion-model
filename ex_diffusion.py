import torch
from diffusers import StableDiffusionPipeline
import uuid
from pathlib import Path
from sd.sd_pipeline import use_sd

def generate_diagram(prompt: str):
    image = use_sd(prompt, ALLOW_MPS=True)
    filename = f"{uuid.uuid4()}.png"
    filepath = Path("generated") / filename
    filepath.parent.mkdir(exist_ok=True)
    image.save(filepath)
    return str(filepath)
