#!/bin/bash
#SBATCH --job-name=trajectory_inference
#SBATCH --time=24:00:00
#SBATCH --environment=trajectory-inference-env
#SBATCH --nodes=1
#SBATCH --account=a03
# #SBATCH --reservation=sai-a03
#SBATCH --output=./logs_slurm/traj_%j.log
#SBATCH --open-mode=append

NUM_GPUS=4
FILE_PATHS_LIST="./output/file_list_node_${NODE_IDX}.txt"
EXP_NAME="traj_node_${NODE_IDX}"

srun python scripts/geocalib_inference.py \
    --file-list $FILE_PATHS_LIST \
    --log-dir "./logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 4 \
    --batch-size 50 \
    --num-workers 24 \
    --camera-model "pinhole" \
    --exp-name $EXP_NAME \
    --no-profiler

srun python scripts/depthanything_inference.py \
    --file-list $FILE_PATHS_LIST \
    --weights-dir "/store/cscs/swissai/a03/weights" \
    --log-dir "./logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 4 \
    --buffer-size 2048 \
    --batch-size 64 \
    --num-workers 24 \
    --encoder "vits" \
    --exp-name $EXP_NAME \
    --no-profiler

srun python scripts/droidslam_inference.py \
    --file-list $FILE_PATHS_LIST \
    --weights "/store/cscs/swissai/a03/weights/droid.pth" \
    --log-dir "./logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 4 \
    --trajectory-length 4000 \
    --trajectory-overlap 100 \
    --min-trajectory-length 100 \
    --num-workers 24 \
    --exp-name $EXP_NAME \
    --no-profiler

echo “”
echo “################################################################”
echo “@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@”
date
echo “@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@”
echo “################################################################”
