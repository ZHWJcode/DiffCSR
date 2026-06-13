import os
import glob
import math
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as f

from diffcsr import DiffCSR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

def diffcsr(args):
    # Initialize the model
    model = DiffCSR_eval(args)
    model.set_eval()

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        # Ensure the input image is a multiple of 8
        input_image_ori = Image.open(image_name).convert('RGB')
        input_image = transforms.ToTensor()(input_image_ori)
        input_image = F.interpolate(input_image.unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False)

        resize_h, resize_w = input_image.shape[2], input_image.shape[3]
        pad_h = (math.ceil(resize_h / 64) * 64) - resize_h
        pad_w = (math.ceil(resize_w / 64) * 64) - resize_w
        input_image = F.pad(input_image.clamp(0, 1), (0, pad_w, 0, pad_h), mode='reflect')

        bname = os.path.basename(image_name)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = input_image.cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image[:, :, :resize_h, :resize_w]
        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image_ori)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image_ori)

        output_pil.save(os.path.join(args.output_dir, bname))

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='input_dir', help="path to the input image")
    parser.add_argument('--output_dir', '-o', type=str, default='output_dir', help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='the path of sd21_base')
    parser.add_argument('--pretrained_path', type=str, default='the path of pkl', help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="no")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--mixed_precision", type=str, default="fp32")
    parser.add_argument("--default", action="store_true", help="use default or adjustale setting?") 

    args = parser.parse_args()

    # Call the processing function
    diffcsr(args)
