import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

PROMPTS_FILE = '/Users/krishnaarora/Desktop/PRO/2/prompts.txt'

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    with torch.no_grad():
        output = pipeline(prompt)
        image = output.images[0] 
    return image

def main():
    with open(PROMPTS_FILE, 'r') as file:
        prompts = file.readlines()
    
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if prompt:
            print(f"Generating image for prompt: {prompt}")
            image = generate_image(prompt)
            
            image_path = f'/Users/krishnaarora/Desktop/PRO/generated_image_{i+1}.png'
            image.save(image_path)
            print(f"Image saved at {image_path}")

if __name__ == "__main__":
    main()
