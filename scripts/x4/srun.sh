# !/bin/bash
set -x

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
PATH_TO_SFT_DATA=data
PATH_TO_PRETRAINED_PROJECTOR=checkpoints/pretrain_weighted
DATA=$1

python -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_mem_v4.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --mm_vision_sample_feature True \
    --mm_vision_sample_num 3 \
    --version v1 \
    --data_path $PATH_TO_SFT_DATA/$DATA \
    --image_folder $PATH_TO_SFT_DATA \
    --vision_tower "clip-448;convnext-1024;det-1024;pix2struct-1024" \
    --pretrain_mm_mlp_adapter $PATH_TO_PRETRAINED_PROJECTOR/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --visualize True