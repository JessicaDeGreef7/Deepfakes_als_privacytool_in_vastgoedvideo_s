#import torch
#from diffusers import StableDiffusion3Pipeline

#pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
#pipe = pipe.to("cuda")

#prompt = "a man with blonde hair and beard"

#image = pipe(
#    prompt,
#    num_inference_steps=40,
#    guidance_scale=4.5,
#).images[0]
#image.save("capybara.png")

from diffusers import StableDiffusionPipeline
import torch

model_path = "Sourabh2/Human_Face_generator"
model_path = "stablediffusionapi/realistic-vision-v13"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

gender = "woman"
age = "10"
eye_color = "green"
hair_color = "blonde"
hair_style = "short"
facial_hair = "nothing"
glasses = "no"


if glasses == "yes":
    prompt = "a" + gender + " of the age of " + age +" with "+ hair_style + ", " + hair_color + " hair and " + eye_color + " eyes with " + facial_hair + " as facial hair and glasses"
else:
    prompt = "a" + gender + " of the age of " + age +" with "+ hair_style + ", " + hair_color + " hair and " + eye_color + " eyes with " + facial_hair + " as facial hair high quality photography"


image = pipe(prompt=prompt).images[0]
image.save("face4.png")