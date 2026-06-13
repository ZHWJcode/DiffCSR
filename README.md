<div align="center">
<h2>A Smooth Decoupling Strategy based on Shared Prior for Compressed Image Super-Resolution</h2>

<a href="#"><img src="https://img.shields.io/badge/Paper-PDF-red"></a>
<a href="#"><img src="https://img.shields.io/badge/Code-DiffCSR-blue"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>

Wenjian Zhang<sup>1</sup> | Jiawei Wu<sup>1</sup> | Zhi Jin<sup>1,2,3,*</sup>

<sup>1</sup>School of Intelligent Systems Engineering, Shenzhen Campus of Sun Yat-sen University, Shenzhen, China  
<sup>2</sup>Guangdong Provincial Key Laboratory of Fire Science and Intelligent Emergency Technology, Shenzhen, China  
<sup>3</sup>Guangdong Provincial Key Laboratory of Robotics and Digital Intelligent Manufacturing Technology, Guangzhou, China  
<sup>*</sup>Corresponding author
</div>

## Update

- Code and checkpoints will be organized for public release.
- This repository contains the implementation of DiffCSR for compressed image super-resolution.

## TODO

- [x] Release training and testing code
- [ ] Release pretrained DiffCSR checkpoints
- [ ] Release visual comparison assets
- [ ] Add paper and project links after publication

## Abstract

Compressed Image Super-Resolution (CSR) aims to recover high-resolution images from inputs degraded by both downsampling and lossy compression. Existing end-to-end methods often overlook that downsampling blur aggravates high-frequency information loss during quantization, while direct cascaded restoration pipelines introduce extra training cost and error accumulation.

We propose DiffCSR, a diffusion-based CSR framework with a smooth decoupling strategy built on shared generative priors. DiffCSR decomposes CSR into two progressive sub-tasks: Compression Artifact Removal (CAR) and Super-Resolution (SR). Two task-specific LoRA modules are trained within a shared Stable Diffusion prior, enabling a smoother transition between the two stages. To improve structural fidelity, we further introduce Progressive Structure Modulation (PSM), which injects structure guidance from the artifact-free latent representation into the SR stage.

Extensive experiments on the UCSR benchmark show that DiffCSR achieves superior perceptual quality under diverse compression settings, producing fewer artifacts and richer textures than existing CSR and cascaded restoration methods.

## Highlights

- Diffusion priors for CSR: DiffCSR introduces pretrained diffusion model priors to compressed image super-resolution.
- Smooth task decoupling: CSR is decomposed into CAR and SR with two task-specific LoRA modules in a shared semantic space.
- Progressive Structure Modulation: PSM constrains texture generation and improves structural consistency across stages.
- Strong perceptual restoration: DiffCSR improves perceptual metrics including CLIPIQA, MUSIQ, MANIQA, LPIPS, DISTS, NIQE, and FID on UCSR benchmarks.

## Framework Overview

The overall pipeline contains two progressive restoration stages:

1. CAR stage: removes compression artifacts and maps compressed low-resolution images to artifact-free latent representations.
2. SR stage: builds on the CAR output and generates high-frequency details for 4x super-resolution.
3. PSM: uses structural information from the CAR latent to modulate UNet up-block features during the SR stage.

<!-- Add the paper framework figure here after preparing GitHub assets, for example:
<p align="center">
  <img src="assets/framework.png" alt="DiffCSR framework" width="850">
</p>
-->

## Visual Results

Example visualization files are provided under `data/visualization/`. After preparing the assets, you can update this section with image sliders or direct comparisons.

### Progressive Restoration

<p align="center">
  <img src="data/visualization/diffusion/0001_jpeg_10.png" height="180">
  <img src="data/visualization/diffusion/0001_jpeg_10_s25.png" height="180">
  <img src="data/visualization/diffusion/0001_jpeg_10_s50.png" height="180">
  <img src="data/visualization/diffusion/0001_jpeg_10_s75.png" height="180">
</p>

### Compression-Aware Prompt Visualization

<p align="center">
  <img src="data/visualization/clip/CAPM_diffcsr_65001pth.svg" alt="Compression-aware prompt visualization" width="650">
</p>

## Dependencies and Installation

```bash
git clone https://github.com/your-name/DiffCSR.git
cd DiffCSR

conda create -n diffcsr python=3.10
conda activate diffcsr
pip install --upgrade pip
pip install -r requirements.txt
```

You can also install from the provided conda environment file:

```bash
conda env create -f environment.yaml
conda activate s3diff
```

This project depends on PyTorch, Diffusers, PEFT, Accelerate, LPIPS, PyIQA, OpenCLIP, and xFormers. We recommend using CUDA-enabled GPUs for both training and inference.

## Pretrained Models

Please prepare the following models before training or testing:

- Stable Diffusion 2.1 base: download from Hugging Face and set `--pretrained_model_path`.
- RAM image tagging model: place `ram_swin_large_14m.pth` under the RAM model path used in `src/train_universal_v9.py`.
- CA-CLIP model: set `--caclip_path` for compression-aware prompt extraction.
- DiffCSR checkpoint: set the checkpoint path in `src/test_universal_v9.py` before inference.

Recommended local structure:

```text
pretrained/
  stable-diffusion-2-1-base/
  ram_swin_large_14m.pth
  caclip_model_best.pkl
checkpoints/
  diffcsr.pkl
```

## Dataset Preparation

DiffCSR is trained and evaluated on the UCSR benchmark. The test split includes BSD100, Urban100, and Manga109, and the code also supports Set5 and Set14.

Expected benchmark organization:

```text
UCSR/
  Train/
    ...
  Test/
    BSD100/
      HR/
      LR_JPEG/10/
      LR_JPEG/40/
      LR_PSNR/2/
      LR_HIFI/high/
    Urban100/
    Manga109/
```

Update dataset paths in:

- `src/csr_dataset.py`
- `src/dataset/diffusion/train_diff/paths_label.txt`
- `src/dataset/diffusion/test_diff/*_paths_label.txt`
- `configs/diffusion.yaml` if you use config-driven loading

## Quick Inference

The current inference script contains several local paths at the top of `src/test_universal_v9.py`. Before running, update:

- `clip_path`
- `pretrained_model_path`
- `model_path`
- `save_dir`
- `TestDataset(test_dataset=..., compre_type=..., qf=...)`

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python src/test_universal_v9.py \
  --align_method adain
```

Supported color alignment modes:

```bash
--align_method adain
--align_method wavelet
--align_method nofix
```

For batch evaluation, modify the `test_sets` list in `src/test_universal_v9.py`. Output images will be saved to the configured `save_dir`.

## Training

### Step 1: Prepare Pretrained Models

Download Stable Diffusion 2.1 base, RAM, and the CA-CLIP checkpoint. Update the corresponding paths in the command below.

### Step 2: Prepare Training Data

Prepare paired CSR training samples and update `src/dataset/diffusion/train_diff/paths_label.txt`. Each training item should provide:

- compressed low-resolution image
- intermediate artifact-free low-resolution target for CAR
- high-resolution ground truth for SR
- compression label or compression-aware prompt input

### Step 3: Train DiffCSR

```bash
accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" \
  --main_process_port 29300 src/train_universal_v9.py \
  --pretrained_model_path="pretrained/stable-diffusion-2-1-base" \
  --pretrained_model_path_csd="pretrained/stable-diffusion-2-1-base" \
  --caclip_path="pretrained/caclip_model_best.pkl" \
  --output_dir="experiments/diffcsr" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --max_train_steps=80000 \
  --sr_steps=30000 \
  --checkpointing_steps=1000 \
  --eval_freq=2500 \
  --lora_rank_unet_cr=4 \
  --lora_rank_unet_sr=4 \
  --lambda_l=1.0 \
  --lambda_perc=2.0 \
  --lambda_csd=1.0 \
  --enable_xformers_memory_efficient_attention
```

The script first optimizes the CAR LoRA. When `global_step == sr_steps`, it switches to the SR LoRA and activates the SR training objective with PSM.

## Evaluation

DiffCSR is evaluated with fidelity and perceptual metrics:

- Fidelity: PSNR, SSIM
- Full-reference perceptual metrics: LPIPS, DISTS, FID
- No-reference perceptual metrics: NIQE, CLIPIQA, MUSIQ, MANIQA

You can compute metrics with:

```bash
python calculate_metrics.py
```

Update the input and reference folders in the script before running.

## Results

On the UCSR benchmark, DiffCSR demonstrates stronger perceptual restoration quality under JPEG, learned image compression, and HIFI compression settings. It is especially effective when compression and downsampling cause severe high-frequency loss, where conventional end-to-end methods tend to produce over-smoothed outputs and cascaded approaches suffer from error propagation.

## Citation

If this work is helpful to your research, please consider citing:

```bibtex
@misc{zhang2026diffcsr,
  title  = {A Smooth Decoupling Strategy based on Shared Prior for Compressed Image Super-Resolution},
  author = {Zhang, Wenjian and Wu, Jiawei and Jin, Zhi},
  year   = {2026}
}
```

Please update the BibTeX entry with the final venue and page information after publication.

## Acknowledgement

This project builds on the progress of diffusion-based image restoration and super-resolution methods, including Stable Diffusion, OSEDiff, PiSA-SR, SeeSR, RAM, and related compressed image restoration benchmarks. We thank the authors for their excellent open-source contributions.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contact

For questions, please contact:

- Wenjian Zhang: zhangwj289@mail2.sysu.edu.cn
- Jiawei Wu: wujw97@mail2.sysu.edu.cn
- Zhi Jin: jinzh26@mail.sysu.edu.cn
