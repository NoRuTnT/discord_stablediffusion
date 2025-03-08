from models.normal_model import normal_pipe, normal_img_pipe
from models.lora_model import lora_pipe
from PIL import Image
import torch

def generate_image(prompt: str, mode: str, image: Image = None, strength: float = 0.10):

    if image:
        if mode == "lora":
            pipe = lora_pipe
        else:
            pipe = normal_img_pipe

        # print("UNet 로드 장치:", pipe.unet.device)
        # print("VAE 로드 장치:", pipe.vae.device)
        # print("텍스트 인코더 로드 장치:", pipe.text_encoder.device)

        init_image = image.convert("RGB").resize((512, 512))
        result = pipe(prompt, image=init_image, strength=strength, guidance_scale=5.0, num_inference_steps=20).images[0]
    else:
        pipe = normal_pipe
        # print("UNet 로드 장치:", pipe.unet.device)
        # print("VAE 로드 장치:", pipe.vae.device)
        # print("텍스트 인코더 로드 장치:", pipe.text_encoder.device)

        result = pipe(prompt, num_inference_steps=20).images[0]

    return result