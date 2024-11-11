#!/bin/bash

cd $FILE_DIR
find ~+ -type f -name "*.h5" > ./file_list.txt


cd $SCRATCH/trajectory_inference
python extraction/generate_trajectory.py \
    --file-list "./file_list.txt" \
    --replace-from "part/of/filepath/to/replace" \
    --replace-to "what/to/replace/with" \
    --encoder "vitl" \
    --weights-dir "/capstor/scratch/cscs/pmartell/trajectory_inference/weights"

