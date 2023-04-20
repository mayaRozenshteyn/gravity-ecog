import os
import sys

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import PIL
import numpy as np
import torch
import torchvision.transforms as T
import cv2

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def extract_image_embeddings(cap):
    ret, frame = cap.read()
    
    i = 0
    while ret:
        
        if (i == 5760):
            # obtain image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = PIL.Image.fromarray(img)

            # preprocess image
            init_image = preprocess_image(im_pil)

            # encode the init image into latents and scale the latents
            init_latent_dist = vae.encode(init_image).latent_dist # self.
            # https://pytorch.org/docs/stable/generated/torch.Generator.html
            init_latents = init_latent_dist.sample(generator=None) #generator=generator 
            init_latents = 0.18215 * init_latents

            torch.save(init_latents, os.path.join(sys.path[0], "Grav2AutoEnc/tensor" + str(i) + ".pt"))

        ret, frame = cap.read()
        i = i + 1
        
    cap.release()
    cv2.destroyAllWindows()
    
cap1 = cv2.VideoCapture('/scratch/gpfs/mayaar/GravityECoG/sourcedata/grav2.mp4')
extract_image_embeddings(cap1)