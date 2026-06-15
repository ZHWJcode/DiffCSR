import os
import sys
import time
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook
from arch_util import PSM, Normalize2

import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None

import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()

class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float32),  # dtype=float32
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float32),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)
            
        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()


def initialize_unet(rank_cr, rank_sr, return_lora_module_names=False, pretrained_model_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_cr, l_target_modules_decoder_cr, l_modules_others_cr = [], [], []
    l_target_modules_encoder_sr, l_target_modules_decoder_sr, l_modules_others_sr = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_cr.append(n.replace(".weight",""))
                l_target_modules_encoder_sr.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_cr.append(n.replace(".weight",""))
                l_target_modules_decoder_sr.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others_cr.append(n.replace(".weight",""))
                l_modules_others_sr.append(n.replace(".weight",""))
                break

    lora_conf_encoder_cr = LoraConfig(r=rank_cr, init_lora_weights="gaussian", target_modules=l_target_modules_encoder_cr)
    lora_conf_decoder_cr = LoraConfig(r=rank_cr, init_lora_weights="gaussian", target_modules=l_target_modules_decoder_cr)
    lora_conf_others_cr = LoraConfig(r=rank_cr, init_lora_weights="gaussian", target_modules=l_modules_others_cr)
    lora_conf_encoder_sr = LoraConfig(r=rank_sr, init_lora_weights="gaussian", target_modules=l_target_modules_encoder_sr)
    lora_conf_decoder_sr = LoraConfig(r=rank_sr, init_lora_weights="gaussian", target_modules=l_target_modules_decoder_sr)
    lora_conf_others_sr = LoraConfig(r=rank_sr, init_lora_weights="gaussian", target_modules=l_modules_others_sr)

    unet.add_adapter(lora_conf_encoder_cr, adapter_name="default_encoder_cr")
    unet.add_adapter(lora_conf_decoder_cr, adapter_name="default_decoder_cr")
    unet.add_adapter(lora_conf_others_cr, adapter_name="default_others_cr")
    unet.add_adapter(lora_conf_encoder_sr, adapter_name="default_encoder_sr")
    unet.add_adapter(lora_conf_decoder_sr, adapter_name="default_decoder_sr")
    unet.add_adapter(lora_conf_others_sr, adapter_name="default_others_sr")

    if return_lora_module_names:
        return unet, l_target_modules_encoder_cr, l_target_modules_decoder_cr, l_modules_others_cr, l_target_modules_encoder_sr, l_target_modules_decoder_sr, l_modules_others_sr
    else:
        return unet

class DiffCSR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder_cr, lora_unet_modules_decoder_cr, lora_unet_others_cr, \
                lora_unet_modules_encoder_sr, lora_unet_modules_decoder_sr, lora_unet_others_sr, =\
                    initialize_unet(rank_cr=args.lora_rank_unet_cr, rank_sr=args.lora_rank_unet_sr, pretrained_model_path=args.pretrained_model_path, return_lora_module_names=True)
            
            self.lora_rank_unet_cr = args.lora_rank_unet_cr
            self.lora_rank_unet_sr = args.lora_rank_unet_sr
            self.lora_unet_modules_encoder_cr, self.lora_unet_modules_decoder_cr, self.lora_unet_others_cr, \
                self.lora_unet_modules_encoder_sr, self.lora_unet_modules_decoder_sr, self.lora_unet_others_sr= \
                lora_unet_modules_encoder_cr, lora_unet_modules_decoder_cr, lora_unet_others_cr, \
                    lora_unet_modules_encoder_sr, lora_unet_modules_decoder_sr, lora_unet_others_sr
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = read_yaml(stage1_yaml)
            stage1_args = SimpleNamespace(**stage1_args)
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet_cr = stage1_args.lora_rank_unet_cr
            self.lora_rank_unet_sr = stage1_args.lora_rank_unet_sr
            pisasr = torch.load(args.resume_ckpt)
            self.load_ckpt_from_state_dict(pisasr)
        # unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')
        self.psm = PSM().to("cuda")
        self.register_hook_unet()

        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

    def set_train_cr(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "cr" in n:
                _p.requires_grad = True
            if "sr" in n:
                _p.requires_grad = True
        
        self.psm.train()
        self.psm.requires_grad_(True)
    
    def set_train_sr(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sr" in n:
                _p.requires_grad = True
            if "cr" in n:
                _p.requires_grad = False
        
        self.psm.train()
        self.psm.requires_grad_(True)

    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        self.lora_conf_encoder_cr = LoraConfig(r=sd["lora_rank_unet_cr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_cr"])
        self.lora_conf_decoder_cr = LoraConfig(r=sd["lora_rank_unet_cr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_cr"])
        self.lora_conf_others_cr = LoraConfig(r=sd["lora_rank_unet_cr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_cr"])

        self.lora_conf_encoder_sr = LoraConfig(r=sd["lora_rank_unet_sr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sr"])
        self.lora_conf_decoder_sr = LoraConfig(r=sd["lora_rank_unet_sr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sr"])
        self.lora_conf_others_sr = LoraConfig(r=sd["lora_rank_unet_sr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sr"])

        self.unet.add_adapter(self.lora_conf_encoder_cr, adapter_name="default_encoder_cr")
        self.unet.add_adapter(self.lora_conf_decoder_cr, adapter_name="default_decoder_cr")
        self.unet.add_adapter(self.lora_conf_others_cr, adapter_name="default_others_cr")

        self.unet.add_adapter(self.lora_conf_encoder_sr, adapter_name="default_encoder_sr")
        self.unet.add_adapter(self.lora_conf_decoder_sr, adapter_name="default_decoder_sr")
        self.unet.add_adapter(self.lora_conf_others_sr, adapter_name="default_others_sr")

        self.lora_unet_modules_encoder_cr, self.lora_unet_modules_decoder_cr, self.lora_unet_others_cr, \
        self.lora_unet_modules_encoder_sr, self.lora_unet_modules_decoder_sr, self.lora_unet_others_sr= \
        sd["unet_lora_encoder_modules_cr"], sd["unet_lora_decoder_modules_cr"], sd["unet_lora_others_modules_cr"], \
            sd["unet_lora_encoder_modules_sr"], sd["unet_lora_decoder_modules_sr"], sd["unet_lora_others_modules_sr"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        self.psm.load_state_dict(sd["psm"])

    def register_hook_unet(self):
        #  register hook
        self.norm1 = Normalize2(in_channels=1280); self.norm2 = Normalize2(in_channels=640); self.norm3 = Normalize2(in_channels=320)
        self.hook_guidance = None; self.modulation = None
        if len(self.unet.up_blocks) == 4:
            self.unet.up_blocks[1].register_forward_hook(self.hook_guidance_for_sr)
            self.unet.up_blocks[2].register_forward_hook(self.hook_guidance_for_sr)
            self.unet.up_blocks[3].register_forward_hook(self.hook_guidance_for_sr)
        else:
            self.unet.up_blocks[-1].register_forward_hook(self.hook_guidance_for_sr)
    
    def hook_guidance_for_sr(self, module, args, output):
        if self.modulation is not None:
            modulation = self.modulation

            if output.shape[1] == modulation['scale1'].shape[1]:  # (chans=1280)
                output_norm = self.norm1(output)
                modified_output = output_norm * (1 + modulation['scale1']) + modulation['shift1']
            elif output.shape[1] == modulation['scale2'].shape[1]:  # (chans=640)
                output_norm = self.norm2(output)
                modified_output = output_norm * (1 + modulation['scale2']) + modulation['shift2']
            elif output.shape[1] == modulation['scale3'].shape[1]:  # (chans=320)
                output_norm = self.norm3(output)
                modified_output = output_norm * (1 + modulation['scale3']) + modulation['shift3']
            return modified_output

        return output

    # Adopted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def forward(self, c_t, c_tgt, stage=None, batch=None, args=None):

        bs = c_t.shape[0]
        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        null_prompt_embeds = self.encode_prompt(batch["null_prompt"])

        if random.random() < args.null_text_ratio:
            pos_caption_enc = null_prompt_embeds
        else:
            pos_caption_enc = prompt_embeds

        x_denoised = None
        if stage == 'cr':
            self.hook_guidance = None; self.modulation = None
            z_cr = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=pos_caption_enc.to(torch.float32),).sample
            x_denoised = encoded_control - z_cr
        elif stage == 'sr':
            with torch.no_grad():
                self.unet.set_adapter(["default_encoder_cr", "default_decoder_cr", "default_others_cr"])
                self.hook_guidance = None; self.modulation = None
                z_cr = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=pos_caption_enc.to(torch.float32),).sample
                x_denoised = encoded_control - z_cr

            self.hook_guidance = x_denoised
            self.modulation = self.psm(z_guidance=self.hook_guidance)
            self.unet.set_adapter(["default_encoder_sr", "default_decoder_sr", "default_others_sr"])
            z_sr = self.unet(x_denoised, self.timesteps1, encoder_hidden_states=pos_caption_enc.to(torch.float32),).sample
            x_denoised = x_denoised - z_sr
            self.hook_guidance = None; self.modulation = None
        
        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_cr"], sd["unet_lora_decoder_modules_cr"], sd["unet_lora_others_modules_cr"] =\
            self.lora_unet_modules_encoder_cr, self.lora_unet_modules_decoder_cr, self.lora_unet_others_cr
        sd["unet_lora_encoder_modules_sr"], sd["unet_lora_decoder_modules_sr"], sd["unet_lora_others_modules_sr"] =\
            self.lora_unet_modules_encoder_sr, self.lora_unet_modules_decoder_sr, self.lora_unet_others_sr
        sd["lora_rank_unet_cr"] = self.lora_rank_unet_cr
        sd["lora_rank_unet_sr"] = self.lora_rank_unet_sr
        sd["psm"] = self.psm.state_dict()
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        torch.save(sd, outf)


class DiffCSR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
        self.psm = PSM().to(self.device)
        # register hook
        self.register_hook_unet()

        # Load pretrained weights
        self._load_pretrained_weights(args.pretrained_path)

        # Initialize VAE tiling
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        # Move models to device and precision
        self._move_models_to_device_and_dtype()

        # Set timesteps
        self.timesteps1 = torch.tensor([1], device=self.device).long()

    def _get_dtype(self, precision):
        """Get the appropriate data type based on precision."""
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _move_models_to_device_and_dtype(self):
        """Move models to the correct device and precision."""
        for model in [self.vae, self.unet, self.text_encoder, self.psm]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights and initialize LoRA adapters."""
        sd = torch.load(pretrained_path, map_location=self.device)  # change
        self._load_and_save_ckpt_from_state_dict(sd)

    def _load_and_save_ckpt_from_state_dict(self, sd):
        """Load checkpoint and initialize LoRA adapters."""
        # Define LoRA configurations
        self.lora_conf_encoder_cr = LoraConfig(r=sd["lora_rank_unet_cr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_cr"])
        self.lora_conf_decoder_cr = LoraConfig(r=sd["lora_rank_unet_cr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_cr"])
        self.lora_conf_others_cr = LoraConfig(r=sd["lora_rank_unet_cr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_cr"])

        self.lora_conf_encoder_sr = LoraConfig(r=sd["lora_rank_unet_sr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sr"])
        self.lora_conf_decoder_sr = LoraConfig(r=sd["lora_rank_unet_sr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sr"])
        self.lora_conf_others_sr = LoraConfig(r=sd["lora_rank_unet_sr"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sr"])

        # Add and load adapters
        self.unet.add_adapter(self.lora_conf_encoder_cr, adapter_name="default_encoder_cr")
        self.unet.add_adapter(self.lora_conf_decoder_cr, adapter_name="default_decoder_cr")
        self.unet.add_adapter(self.lora_conf_others_cr, adapter_name="default_others_cr")

        for name, param in self.unet.named_parameters():
            if "cr" in name:
                param.data.copy_(sd["state_dict_unet"][name])
        
        # Add srantic adapters
        self.unet.add_adapter(self.lora_conf_encoder_sr, adapter_name="default_encoder_sr")
        self.unet.add_adapter(self.lora_conf_decoder_sr, adapter_name="default_decoder_sr")
        self.unet.add_adapter(self.lora_conf_others_sr, adapter_name="default_others_sr")
        
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        self.psm.load_state_dict(sd["psm"])

    def set_eval(self):
        """Set models to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.psm.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.psm.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def count_parameters(self, model):
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters()) / 1e9

    def register_hook_unet(self):
        #  register Hook
        self.norm1 = Normalize2(in_channels=1280); self.norm2 = Normalize2(in_channels=640); self.norm3 = Normalize2(in_channels=320)
        self.hook_guidance = None; self.modulation = None
        
        if len(self.unet.up_blocks) == 4:
            self.unet.up_blocks[1].register_forward_hook(self.hook_guidance_for_sr)
            self.unet.up_blocks[2].register_forward_hook(self.hook_guidance_for_sr)
            self.unet.up_blocks[3].register_forward_hook(self.hook_guidance_for_sr)
        else:
            self.unet.up_blocks[-1].register_forward_hook(self.hook_guidance_for_sr)
    
    def hook_guidance_for_sr(self, module, args, output):
        if self.modulation is not None:
            modulation = self.modulation
            
            if output.shape[1] == modulation['scale16'].shape[1]:
                output_norm = self.norm1(output)
                modified_output = output_norm * (1 + modulation['scale16']) + modulation['shift16']
            elif output.shape[1] == modulation['scale32'].shape[1]:
                output_norm = self.norm2(output)
                modified_output = output_norm * (1 + modulation['scale32']) + modulation['shift32']
            elif output.shape[1] == modulation['scale64'].shape[1]:
                output_norm = self.norm3(output)
                modified_output = output_norm * (1 + modulation['scale64']) + modulation['shift64']
            return modified_output

        return output

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        """Forward pass for inference."""
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # Two stage csr
        set_weights_and_activate_adapters(self.unet, ["default_encoder_cr", "default_decoder_cr", "default_others_cr"], [1.0, 1.0, 1.0])
        self.hook_guidance = None; self.modulation = None
        z_cr = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        x_denoised = encoded_control - z_cr

        set_weights_and_activate_adapters(self.unet, ["default_encoder_sr", "default_decoder_sr", "default_others_sr"], [1.0, 1.0, 1.0])
        self.hook_guidance = x_denoised
        self.modulation = self.psm(z_guidance=self.hook_guidance)
        z_sr = self.unet(x_denoised, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        x_denoised = x_denoised - z_sr
        self.hook_guidance = None; self.modulation = None

        # Decode output
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        return total_time, output_image

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        """Initialize VAE with tiled encoding/decoding."""
        encoder, decoder = self.vae.encoder, self.vae.decoder

        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward

        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)