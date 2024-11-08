#!/bin/bash

# Job parameters
JOB_NAME="trajectory_inference"
TIME_LIMIT="12:00:00"
ENVIRONMENT="trajectory-inference-env"
NODES=1
ACCOUNT="a03"
OUTPUT_LOG="./logs_slurm/traj_$(date +%Y%m%d_%H%M%S).log"
WORKDIR="/capstor/scratch/cscs/$USER/trajectory_inference"
MEMORY=460000
NUM_GPUS=4

# Set file paths and experiment names based on node index
NODE_IDX=$1  # Default to 0 if NODE_IDX is not set
FILE_PATHS_LIST="./output/file_list_node_${NODE_IDX}.txt"
EXP_NAME="traj_node_${NODE_IDX}"

echo "Running node index: $NODE_IDX"

# Change to the working directory


# # Run the third script
srun --nodes=$NODES \
     --environment="$ENVIRONMENT" \
     --account=a03 \
     --mem=460000 \
     --ntasks-per-node=1 \
     --time="$TIME_LIMIT" \
     ./traj4.sh $FILE_PATHS_LIST $EXP_NAME

# Completion message
# echo ""
# echo "################################################################"
# echo "@@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
# date
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "################################################################"