from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import torch
from config import BASE_MODEL


def load_normal_model():
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL,torch_dtype=torch.float16).to("cuda")

    return pipe

def load_normal_img_model():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(BASE_MODEL,torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

normal_pipe = load_normal_model()
normal_img_pipe = load_normal_img_model()