
CUDA_VISIBLE_DEVICES="0,1,2,3," accelerate launch train_pisasr.py \
    --pretrained_model_path="preset/models/stable-diffusion-2-1-base" \
    --pretrained_model_path_csd="preset/models/stable-diffusion-2-1-base" \
    --dataset_txt_paths="preset/gt_path.txt" \
    --highquality_dataset_txt_paths="preset/gt_selected_path.txt" \
    --dataset_test_folder="preset/testfolder" \
    --learning_rate=5e-5 \
    --train_batch_size=4 \
    --prob=0.0 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 500 \
    --seed 123 \
    --output_dir="experiments/train-pisasr" \
    --cfg_csd 7.5 \
    --timesteps1 1 \
    --lambda_lpips=2.0 \
    --lambda_l2=1.0 \
    --lambda_csd=1.0 \
    --pix_steps=4000 \
    --lora_rank_unet_pix=4 \
    --lora_rank_unet_sem=4 \
    --min_dm_step_ratio=0.02 \
    --max_dm_step_ratio=0.5 \
    --null_text_ratio=0.5 \
    --align_method="adain" \
    --deg_file_path="params.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True


# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# nohup accelerate launch --num_processes=2 --gpu_ids="6,7" --main_process_port 23001 train_icme.py > train_jpeg10_icme.log 2>&1 &