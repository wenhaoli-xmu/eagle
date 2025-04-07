
# 到对应的脚本里修改以下变量的路径位置
# PATH_TO_SFT_DATA
# PATH_TO_PRETRAINED_PROJECTOR

# v4
# srun --partition INTERN2 --gres "gpu:8" --ntasks-per-node 1 -N 2 --job-name python --time 14-00:00 bash scripts/x4/srun.sh eagle_x4_sft
# v5

bash scripts/x5/srun_sft.sh eagle_x5_sft