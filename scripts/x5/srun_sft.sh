# !/bin/bash
set -x

NAME='test'
NAME="${NAME}_sft"
export WANDB_PROJECT="eagle-slurm"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR

echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

export PYTHONPATH="$(pwd):${PYTHONPATH}"
MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}


PATH_TO_SFT_DATA=data
PATH_TO_PRETRAINED_PROJECTOR=checkpoints/pretrain_normal


python train_mem_v4.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --mm_vision_sample_feature True \
    --mm_vision_sample_num 3 \
    --version v1 \
    --data_path $PATH_TO_SFT_DATA/eagle-1-sft-1_8M.json \
    --image_folder $PATH_TO_SFT_DATA/images \
    --vision_tower "clip-448;convnext-1024;sam-1024;det-1024;pix2struct-1024" \
    --pretrain_mm_mlp_adapter $PATH_TO_PRETRAINED_PROJECTOR/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$NAME \
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
    --run_name ${NAME} 