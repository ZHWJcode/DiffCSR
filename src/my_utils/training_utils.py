import os
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--is_module", default=True)
    parser.add_argument("--tracker_project_name", type=str, default="DiffCSR")

    # args for the loss function
    parser.add_argument("--lambda_perc", default=2.0, type=float)
    parser.add_argument("--lambda_l", default=2.0, type=float)
    parser.add_argument("--lambda_gan", default=0.05, type=float)

    # args for the prompt
    parser.add_argument("--neg_prompt", default="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)
    parser.add_argument("--pos_prompt", default="", type=str)
    # parser.add_argument("--pos_prompt", default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting", type=str)

    # args for the `t` test
    parser.add_argument("--timesteps1", default=1, type=float)
    # details about the model architecture
    parser.add_argument("--pretrained_model_path", default='huggingface/sd21_base')
    # # unet lora setting
    parser.add_argument("--lora_rank_unet_cr", default=4, type=int)
    parser.add_argument("--lora_rank_unet_sr", default=4, type=int)

    # dataset options
    parser.add_argument("--null_text_ratio", default=0.5, type=float)
    parser.add_argument("--prob", default=0.1, type=float)
    parser.add_argument("--resolution_ori", type=int, default=512,)
    parser.add_argument("--resolution_tgt", type=int, default=512,)

    # resume
    parser.add_argument("--resume_ckpt", default=None, type=str)

    # training details
    parser.add_argument("--output_dir", default='output_dir')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=20000,)
    parser.add_argument("--cr_steps", type=int, default=6000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000,)
    parser.add_argument("--eval_freq", type=int, default=2500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--deg_file_path", default="params.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args