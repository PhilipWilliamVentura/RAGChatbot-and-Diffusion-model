import torch
from diffusers import StableDiffusionPipeline
import uuid
from pathlib import Path

# Load model once
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
)
pipe.to("mps")

def generate_diagram(prompt: str):
    image = pipe(prompt).images[0]
    filename = f"{uuid.uuid4()}.png"
    filepath = Path("generated") / filename
    filepath.parent.mkdir(exist_ok=True)
    image.save(filepath)
    return str(filepath)
