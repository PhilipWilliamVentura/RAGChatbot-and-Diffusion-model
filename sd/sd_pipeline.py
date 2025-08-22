from . import model_loader
from . import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
from pathlib import Path

def use_sd(prompt, uncond_prompt="", image_path="", DEVICE='cpu', ALLOW_CUDA=False, ALLOW_MPS=False):
    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = 'cuda'
    elif (torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = 'mps'
    print(f'Using device {DEVICE}')

    BASE_DIR = Path(__file__).parent.parent  # points to RAGChatbot-and-Diffusion-model/
    tokenizer = CLIPTokenizer(
    BASE_DIR / "data" / "vocab.json",
    merges_file= BASE_DIR / "data" / "merges.txt"
    )
    model_file = BASE_DIR / "data" / "v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    # TEXT TO IMAGE

    prompt = prompt # Add prompt here
    uncond_prompt = uncond_prompt # Can use as a negative prompt
    do_cfg = True
    cfg_scale = 11

    ## IMAGE TO IMAGE

    input_image = None
    image_path = image_path # Add image here
    if image_path:
        input_image = Image.open(image_path)
    strength = 0.9

    sampler = 'ddpm'
    num_inference_steps = 50
    seed = 42

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device='cpu',
        tokenizer=tokenizer
    )

    return Image.fromarray(output_image)