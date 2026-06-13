import os
import gc
import lpips
import pyiqa
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

import diffusers
from pathlib import Path
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from diffcsr import CharbonnierLoss, CSDLoss, DiffCSR
from src.my_utils.training_utils import parse_args  
from src.datasets.dataset import pairedDiffusionDataset
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from arch_util import count_trainable_parameters


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)
        writer = SummaryWriter(os.path.join(args.output_dir, "log"))

    net_csr = DiffCSR(args)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_csr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_csr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # init Charbonnier loss
    loss_char = CharbonnierLoss()

    # init Perceptual Loss
    loss_dists = pyiqa.create_metric('dists', device=accelerator.device, as_loss=True)
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # init Gan model
    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    # # set gen adapter
    net_csr.unet.set_adapter(['default_encoder_cr', 'default_decoder_cr', 'default_others_cr'])
    net_csr.set_train_cr() # first to remove degradation

    # calculate the number of trainable parameters
    # lora_psm = count_trainable_parameters(net_csr.psm)
    # lora_all = count_trainable_parameters(net_csr)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_csr.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    for n, _p in net_csr.psm.named_parameters():  # add PSM parameters
        layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    
    # initialize the dataset
    dataset_train = pairedDiffusionDataset(split="train", args=args, txtpath="hifichigh.txt")
    dataset_val = pairedDiffusionDataset(split="test", args=args, txtpath="Urban100_paths_hifichigh.txt")
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # init RAM for text prompt extractor
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='src/ram_pretrain_model/ram_swin_large_14m.pth',
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    net_csr, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_csr, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_lpips = accelerator.prepare(net_lpips)

    net_disc.to(accelerator.device, dtype=weight_dtype)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    stage = "cr"; lambda_l = 1.0; lambda_perc = 1.0; lambda_gan = 0.0
    l_acc = [net_csr, net_disc]

    if args.resume_ckpt is not None:
        args.cr_steps = 1
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]  # range: [-1, 1]
                x_med = batch["med_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{each_caption}, {args.pos_prompt_csd}' for each_caption in caption]

                if stage == "cr":
                    # the optimization target of cr stage
                    x_tgt = x_med
                    
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = net_csr(x_src, x_tgt, stage=stage, batch=batch, args=args)
                if stage == "cr":
                    loss_l = loss_char(x_tgt_pred.float(), x_tgt.float()) * lambda_l
                    loss_perc = loss_dists(x_tgt_pred.float(), x_tgt.float()) * lambda_perc
                else:
                    loss_l = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l
                    loss_perc = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_perc
                loss_rec = loss_l + loss_perc

                accelerator.backward(loss_rec, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # GAN Loss
                """
                Generator loss: fool the discriminator
                """
                # freeze discriminator when computing G's loss
                for p in net_disc.parameters():
                    p.requires_grad = False
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = net_csr(x_src.detach(), x_tgt, stage=stage, batch=batch, args=args)
                loss_gan = net_disc(x_tgt_pred, for_G=True).mean() * lambda_gan
                accelerator.backward(loss_gan)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                """
                Discriminator loss: fake image vs real image
                """
                # unfreeze discrinator parameters for D update
                for p in net_disc.parameters():
                    p.requires_grad = True
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean()* lambda_gan
                accelerator.backward(lossD_real.mean())
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean()* lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake
            
                if accelerator.is_main_process:
                    writer.add_scalar('Loss_l', loss_l.detach().item(), global_step)
                    writer.add_scalar('Loss_perc', loss_perc.detach().item(), global_step)
                    writer.add_scalar('Loss_gan', loss_gan.detach().item(), global_step)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step == args.cr_steps and accelerator.sync_gradients:
                    # begin the super-resolution optimization
                    if args.is_module:
                        net_csr.module.set_train_sr() 
                    else:
                        net_csr.set_train_sr()
                    
                    stage = "sr"; lambda_l = args.lambda_l; lambda_perc = args.lambda_perc; lambda_gan = args.lambda_gan

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_gan"] = loss_gan.detach().item()
                    logs["loss_l"] = loss_l.detach().item()
                    logs["loss_perc"] = loss_perc.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_csr).save_model(outf)

                    # test
                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # get text prompts from LR
                                x_src_ram = ram_transforms(x_src * 0.5 + 0.5)
                                caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                batch_val["prompt"] = caption
                                # forward pass
                                x_tgt_pred, latents_pred, _, _ = accelerator.unwrap_model(net_csr)(x_src, x_tgt, stage=stage,
                                                                                                      batch=batch_val,
                                                                                                      args=args)
                                # save the output
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                else:
                                    pass
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)

                    accelerator.log(logs, step=global_step)
    
    if accelerator.is_main_process:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
