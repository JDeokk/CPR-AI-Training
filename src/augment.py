from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import os
import glob
from natsort import natsorted
from .config import Config

def run_inpainting_augmentation(image_folder, mask_folder, save_folder):
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    print(f"Loading SDXL: {Config.SDXL_MODEL_ID}...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        Config.SDXL_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to(Config.DEVICE)
    
    image_files = natsorted(glob.glob(os.path.join(image_folder, "*")))
    mask_files = natsorted(glob.glob(os.path.join(mask_folder, "*")))
    mask_dict = {os.path.basename(m).split('_', 1)[0]: m for m in mask_files}
    
    gen = torch.Generator(device=Config.DEVICE).manual_seed(Config.SEED)
    
    cnt = 0
    for img_f in image_files:
        key = os.path.basename(img_f).split('_', 1)[0]
        mask_f = mask_dict.get(key)
        
        if mask_f:
            img = load_image(img_f).resize((1024, 1024))
            mask = load_image(mask_f).resize((1024, 1024))
            
            prompt = ["office", "class room", "living room", "outdoor background"][cnt % 4]
            
            res = pipe(
                prompt=prompt, prompt_2="real world",
                negative_prompt="person body, deformed",
                image=img, mask_image=mask,
                guidance_scale=20, num_inference_steps=30, strength=0.99, generator=gen
            ).images[0]
            
            res.save(os.path.join(save_folder, os.path.basename(img_f)))
            cnt += 1
