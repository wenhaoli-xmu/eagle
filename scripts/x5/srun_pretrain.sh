# !/bin/bash
set -x

NAME=$1
NAME="${NAME}_pretrain"
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

PATH_TO_PRETRAINING_DATA=/mnt/hwfile/share_data/xiejingjing/data/LLaVA-Pretrain

python -m torch.distributed.run \
    --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port 25031 \
    train_mem_v4.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --mm_vision_sample_feature True \
    --mm_vision_sample_num 5 \
    --version plain \
    --data_path $PATH_TO_PRETRAINING_DATA/blip_laion_cc_sbu_558k.json \
    --image_folder $PATH_TO_PRETRAINING_DATA/images \
    --vision_tower "clip-448;convnext-1024;sam-1024;det-1024;pix2struct-1024" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${NAME} 