#!/bin/bash

# Job parameters
JOB_NAME="trajectory_inference"
TIME_LIMIT="6:00:00"
ENVIRONMENT="trajectory-inference-env"
NODES=1
ACCOUNT="a03"
OUTPUT_LOG="./logs_slurm/traj_$(date +%Y%m%d_%H%M%S).log"
WORKDIR="/capstor/scratch/cscs/dbrggema/trajectory_inference"
MEMORY=460000
NUM_GPUS=4

# Set file paths and experiment names based on node index
NODE_IDX=$1  # Default to 0 if NODE_IDX is not set
FILE_PATHS_LIST="./output/file_list_node_${NODE_IDX}.txt"
EXP_NAME="traj_node_${NODE_IDX}"

echo "Running node index: $NODE_IDX"

# Change to the working directory


# # Run the third script
for GPU in {0..3} ; do
    srun --nodes=$NODES \
        --environment="$ENVIRONMENT" \
        --account=a03 \
        --mem=460000 \
        --ntasks-per-node=1 \
        --time=6:00:00 \
        ./prep.sh python3 scripts/droidslam_inference.py \
        --file-list "$FILE_PATHS_LIST" \
        --file-list-idx $GPU \
        --replace-from "/store/swissai/a03/datasets" \
        --replace-to "$SCRATCH/datasets" \
        --weights "/capstor/scratch/cscs/pmartell/trajectory_inference/weights/droid.pth" \
        --log-dir "./logs" \
        --num-gpus 1 \
        --num-proc-per-gpu 1 \
        --trajectory-length 1700 \
        --trajectory-overlap 100 \
        --min-trajectory-length 100 \
        --num-workers 24 \
        --exp-name "$EXP_NAME" \
        --no-profiler || true
done

# Completion message
# echo ""
# echo "################################################################"
# echo "@@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
# date
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "################################################################"