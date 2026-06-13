import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from torch.utils import data as data
from src.datasets.realesrgan import RealESRGAN_degradation


class pairedDiffusionDataset(data.Dataset):
    def __init__(self, args,
        split="train",
        txtpath="hifichigh.txt",
        test_dataset="Urban100",
        train_basepath="/home/zhangwenjian/image_resolution/PiSA-SR/src/datasets/diffusion/train_diff",
        test_basepath="/home/zhangwenjian/image_resolution/PiSA-SR/src/datasets/diffusion/test_diff"):
        super(pairedDiffusionDataset, self).__init__()

        self.args = args
        self.split = split
        self.txtpath = txtpath
        self.test_dataset = test_dataset
        self.basepath = train_basepath if split == "train" else test_basepath
        txt_path = os.path.join(self.basepath, txtpath)

        self.images_path = []
        with open(txt_path, 'r', encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    self.images_path.append(parts)
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def resize_patch(self, lq_image_patch):
        """
        patch_size: the size of the lq_image
        """
        min_size = self.args.resolution_tgt  # diffusion model requires at least 512x512 input size
        c, h, w = lq_image_patch.shape
        
        # 计算需要填充的像素
        pad_w = max(0, min_size - w); pad_h = max(0, min_size - h)

        if pad_w > 0 or pad_h > 0:
            np_img = lq_image_patch.cpu().numpy()
            np_img_padded = np.pad(np_img, pad_width=((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            lq_image_patch = torch.from_numpy(np_img_padded).to(lq_image_patch.device)
        
        assert min(lq_image_patch.shape[1], lq_image_patch.shape[2]) >= min_size, \
            f"Image size after padding is still smaller than {min_size}x{min_size}"

        return lq_image_patch

    def __getitem__(self, index):
        line = self.images_path[index]
        lq_image_path = line[0]; med_image_path = line[1]; lq_label = int(line[2])

        if self.split == "train":
            patch_size = 128
            hq_basepath = "/data1/zhangwj/datasets/DF2K"
            image_basepath = "/data1/zhangwj/datasets/UCSR/Train"

            lq_image = Image.open(os.path.join(image_basepath, lq_image_path)).convert("RGB")
            med_image = Image.open(os.path.join(image_basepath, med_image_path)).convert("RGB")
            hq_image = Image.open(os.path.join(hq_basepath, os.path.basename(med_image_path))).convert("RGB")

            # 随机裁剪为 pathc_size=128 的图像块
            w, h = lq_image.size  # return [w, h]
            start_w = random.randint(0, w - patch_size)
            start_h = random.randint(0, h - patch_size)
            
            lq_image = self.preprocess(lq_image)
            lq_image_patch = lq_image[:, start_h:start_h+patch_size, start_w:start_w+patch_size]
            lq_image_patch = torch.nn.functional.interpolate(lq_image_patch.unsqueeze(0), size=(self.args.resolution_tgt, self.args.resolution_tgt), mode='bicubic', align_corners=False)
            lq_image_patch = lq_image_patch.clamp(0, 1).squeeze(0)

            med_image = self.preprocess(med_image)
            med_image_patch = med_image[:, start_h:start_h+patch_size, start_w:start_w+patch_size]
            med_image_patch = torch.nn.functional.interpolate(med_image_patch.unsqueeze(0), size=(self.args.resolution_tgt, self.args.resolution_tgt), mode='bicubic', align_corners=False)
            med_image_patch = med_image_patch.clamp(0, 1).squeeze(0)

            hq_image = self.preprocess(hq_image)
            hq_image_patch = hq_image[:, start_h*4:(start_h+patch_size)*4, start_w*4:(start_w+patch_size)*4]

            # 数据增强
            if random.random() < 0.05:
                lq_image_patch = F.hflip(lq_image_patch)
            if random.random() < 0.05:
                lq_image_patch = F.vflip(lq_image_patch)
        else:
            txtnanme = os.path.splitext(self.txtpath)[0]
            test_dataset = txtnanme.split('_')[0]

            image_basepath = f"/data1/zhangwj/datasets/UCSR/Test/{self.test_dataset}"
            lq_image = Image.open(os.path.join(image_basepath, lq_image_path)).convert("RGB")
            med_image = Image.open(os.path.join(image_basepath, med_image_path)).convert("RGB")
            hq_image = Image.open(os.path.join(image_basepath, "HR", os.path.basename(med_image_path))).convert("RGB")

            lq_image_patch = self.preprocess(lq_image)
            lq_image_patch = lq_image_patch.unsqueeze(0)
            lq_h , lq_w = lq_image_patch.shape[2], lq_image_patch.shape[3]
            lq_image_patch = torch.nn.functional.interpolate(lq_image_patch, size=(lq_h * 4, lq_w * 4), mode='bicubic', align_corners=False)
            new_h = lq_image_patch.shape[2] - lq_image_patch.shape[2] % 8
            new_w = lq_image_patch.shape[3] - lq_image_patch.shape[3] % 8
            lq_image_patch = torch.nn.functional.interpolate(lq_image_patch, size=(new_h, new_w), mode='bicubic', align_corners=False)
            lq_image_patch = lq_image_patch.clamp(0, 1).squeeze(0)

            med_image_patch = self.preprocess(med_image)
            med_image_patch = med_image_patch.unsqueeze(0)
            med_h , med_w = med_image_patch.shape[2], med_image_patch.shape[3]
            med_image_patch = torch.nn.functional.interpolate(med_image_patch, size=(med_h * 4, med_w * 4), mode='bicubic', align_corners=False)
            new_h = med_image_patch.shape[2] - med_image_patch.shape[2] % 8
            new_w = med_image_patch.shape[3] - med_image_patch.shape[3] % 8
            med_image_patch = torch.nn.functional.interpolate(med_image_patch, size=(new_h, new_w), mode='bicubic', align_corners=False)
            med_image_patch = med_image_patch.clamp(0, 1).squeeze(0)

            assert med_h == lq_h and med_w == lq_w, print(f"med image size {med_w}x{med_h} not equal to lq image size {lq_w}x{lq_h}")
            
            hq_image_patch = self.preprocess(hq_image)
        
        # input images scaled to -1, 1
        img_t= F.normalize(lq_image_patch, mean=[0.5], std=[0.5])
        # med images scaled to -1, 1
        med_t = F.normalize(med_image_patch, mean=[0.5], std=[0.5])
        # output images scaled to -1, 1
        output_t= F.normalize(hq_image_patch, mean=[0.5], std=[0.5])

        example = {}
        example["null_prompt"] = ""
        example["neg_prompt"] = self.args.neg_prompt
        example["pos_prompt"] = self.args.pos_prompt

        example["conditioning_pixel_values"] = img_t
        example["med_pixel_values"] = med_t
        example["output_pixel_values"] = output_t
        example["base_name"] = os.path.basename(med_image_path)

        return example
    
    def __len__(self):
        return len(self.images_path)