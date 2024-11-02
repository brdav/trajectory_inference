#!/bin/bash
#SBATCH --job-name=trajectory_inference
#SBATCH --time=6:00:00
#SBATCH --environment=dataprep-venv
#SBATCH --nodes=1
#SBATCH --account=a03
#SBATCH --container-workdir=/capstor/scratch/cscs/pmartell/AutonomousDriving
#SBATCH --reservation=sai-a03
#SBATCH --output=prep.out
#SBATCH --error=prep.err

NUM_GPUS=1
FILE_PATHS_LIST="list.txt"
EXP_NAME="exp"

srun python trajectory_inference/scripts/geocalib_inference.py \
    --file-list $FILE_PATHS_LIST \
    --log-dir "trajectory_inference/logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 3 \
    --batch-size 50 \
    --num-workers 4 \
    --camera-model "pinhole" \
    --exp-name $EXP_NAME \
    --no-profiler

srun python trajectory_inference/scripts/depthanything_inference.py \
    --file-list $FILE_PATHS_LIST \
    --weights-dir "/store/cscs/swissai/a03/weights" \
    --log-dir "trajectory_inference/logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 1 \
    --buffer-size 2048 \
    --batch-size 144 \
    --num-workers 12 \
    --encoder "vits" \
    --exp-name $EXP_NAME \
    --no-profiler

srun python trajectory_inference/scripts/droidslam_inference.py \
    --file-list $FILE_PATHS_LIST \
    --weights "/store/cscs/swissai/a03/weights/droid.pth" \
    --log-dir "trajectory_inference/logs" \
    --num-gpus $NUM_GPUS \
    --num-proc-per-gpu 3 \
    --trajectory-length 4000 \
    --trajectory-overlap 100 \
    --min-trajectory-length 100 \
    --num-workers 4 \
    --exp-name $EXP_NAME \
    --no-profiler

echo “”
echo “################################################################”
echo “@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@”
date
echo “@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@”
echo “################################################################”
