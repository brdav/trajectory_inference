# Trajectory Inference

## Setup

Clone with `--recursive` flag.

In an environment with Python==3.9:

```
pip install -r requirements.txt
# droidslam
pip install torch-scatter
pip install xformers == 0.0.21  # double check
cd droid_trajectory/droid_slam
python setup.py install
# geocalib
python -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
```

Download the V2-Large model from [here](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth).
Download the droid.pth checkpoint from [here](https://github.com/princeton-vl/DROID-SLAM)


## Run GeoCalib

When running on multiple GPUs, change `num-gpus`. Try to get `batch-size 50` by using GPUs with 24 GB memory. If larger GPU, increase `num-proc-per-gpu`.
```
python geocalib_inference_v2.py \
    --exp-name "exp_name" \
    --file-list "list_of_file_paths.txt" \
    --log-dir "./logs" \
    --num-gpus 1 \
    --num-proc-per-gpu 1 \
    --batch-size 50 \
    --num-workers 4 \
    --camera-model "pinhole" \
    --no-profiler
```


## Run DepthAnythingV2

When running on multiple GPUs, change `num-gpus`. Try to max out batchsize and then also pick a large buffer size if possible.
```
python depthanything_inference_v2.py \
    --exp-name "exp_name" \
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


## Run DroidSLAM

Run with `trajectory-length 4000`, takes about 24GB if h5 video has >4000 frames. If larger GPU, increase `num-proc-per-gpu`.
```
python droidslam_inference_v2.py \
    --exp-name "exp_name" \
    --file-list "list_of_file_paths.txt" \
    --weights "/path/to/droid.pth" \
    --log-dir "./logs" \
    --num-gpus 1 \
    --num-proc-per-gpu 1 \
    --trajectory-length 4000 \
    --trajectory-overlap 100 \
    --min-trajectory-length 100 \
    --num-workers 2 \
    --no-profiler
```
