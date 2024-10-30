# GeoCalib

## Setup

In an environment with Python>=3.9:
```
python -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
```

## How to run

When running on multiple GPUs, change `num-gpus`. Try to get batchsize==50 by using GPUs with 24 GB memory.
```
python geocalib_inference.py \
    --exp-name "exp_name" \
    --save-dir "./output" \
    --file-list "list_of_file_paths.txt" \
    --log-dir "./logs" \
    --num-gpus 1 \
    --num-proc-per-gpu 1 \
    --batch-size 50 \
    --num-workers 1 \
    --no-profiler
```


# DepthAnytingV2

## Setup

In an environment with Python>=3.9:
```
cd depth_anything_v2
pip install -r requirements.txt
```

Download the V2-Large model from https://github.com/DepthAnything/Depth-Anything-V2

## How to run

When running on multiple GPUs, change `num-gpus`. Try to max out batchsize and then also pick a large buffer size if possible.
```
python depthanything_inference.py \
    --exp-name "exp_name" \
    --save-dir "./output" \
    --file-list "list_of_file_paths.txt" \
    --weights-dir "/dir/to/checkpoint" \
    --log-dir "./logs" \
    --num-gpus 1 \
    --num-proc-per-gpu 1 \
    --buffer-size 2048 \
    --batch-size 32 \
    --num-workers 8 \
    --no-profiler
```