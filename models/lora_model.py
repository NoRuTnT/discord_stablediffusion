from diffusers import StableDiffusionPipeline
import torch
from config import BASE_MODEL
from config import BASE_MODEL, LORA_MODEL_PATH

def load_lora_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_lora_weights(LORA_MODEL_PATH)
    return pipe

lora_pipe = load_lora_model()