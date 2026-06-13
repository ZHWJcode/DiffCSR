import os
import gc
import sys
import tqdm
import math
import glob
import lpips
import pyiqa
import argparse
import numpy as np
from PIL import Image
sys.path.append(os.getcwd())

import torch
import transformers
import torch.utils.checkpoint
import torch.nn.functional as F
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from torch.utils import data as data

from pathlib import Path
from src.utils import util_image

test_dataset_name = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]


class Testdataset(data.Dataset):
    def __init__(self):
        super(Testdataset, self).__init__()

        self.datset_name = test_dataset_name[2]
        self.csr_dir = "DiffCSR/HST/images"
        self.gt_dir = "datasets/UCSR/Test"

        self.gt_path = sorted(glob.glob(f'{self.hr_dir}/{self.datset_name}/HR/*.png'))

    def __getitem__(self, index):
        gt_path = self.hr_path[index]
        gt_basename = os.path.basename(gt_path).split('.')[0]
        lr_path = os.path.join(self.csr_dir, self.datset_name, gt_basename)

        return {"lr_path": lr_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.gt_path)


def evaluate(in_path, ref_path, ntest):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    metric_paired_dict = {}

    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()

    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None:
            ref_path_list = ref_path_list[:ntest]

        metric_paired_dict["psnr"] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"] = pyiqa.create_metric('lpips').to(device)
        metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ssim"] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)

    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None:
        lr_path_list = lr_path_list[:ntest]

    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        _in_path = lr_path_list[i]
        _ref_path = ref_path_list[i] if ref_path_list is not None else None

        im_in = util_image.imread(_in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()  # 1 x c x h x w
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()

        if ref_path is not None:
            im_ref = util_image.imread(_ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()
            if im_in_tensor.shape[-2:] != im_ref_tensor.shape[-2:]:
                im_in_tensor = F.interpolate(im_in_tensor, size=im_ref_tensor.shape[-2:], mode='bilinear', align_corners=False)
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()

    if ref_path is not None:
        fid_metric = pyiqa.create_metric('fid')
        result['fid'] = fid_metric(in_path, ref_path)

    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res / len(lr_path_list):.5f}")
            print_results.append(f"{key}: {res / len(lr_path_list):.5f}")

    return print_results


def main():
    for i in range(0, 5):  # change
        # --- Load inference results ---
        dataset_name = test_dataset_name[i]
        print("Test dataset name is {}".format(dataset_name))
        csr_basepath = "DiffCSR/UCIP/images/JPEG40"
        ref_basepath = "datasets/UCSR/Test"

        csr_path = os.path.join(csr_basepath, dataset_name)
        ref_path = os.path.join(ref_basepath, dataset_name, "HR")

        # --- Compute quality metrics ---
        print_results = evaluate(csr_path, ref_path, None)
        out_t = os.path.join(csr_basepath, f'{dataset_name}_results.txt')
        with open(out_t, 'w', encoding='utf-8') as f:
            for item in print_results:
                f.write(f"{item}\n")


if __name__ == "__main__":
    main()
