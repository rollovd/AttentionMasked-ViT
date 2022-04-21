import wandb
import torch
from PIL import Image

def _wandb_images(number_of_images, batch_of_images):
    images_wandb = [
            Image.fromarray((image * 255).type(torch.uint8).permute(1, 2, 0).cpu().detach().numpy(), 'RGB') \
            for image in batch_of_images[:number_of_images]]

    wandb.log({'train/crop-images': [wandb.Image(image) for image in images_wandb]})