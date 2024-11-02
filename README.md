# Trajectory Inference

## Setup

Clone the repository:
```
git clone --recursive git@github.com:brdav/trajectory_inference.git
```

In an environment with Python>=3.9 and CUDA 12.1:
```
pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install opencv-python h5py scipy tensorboard
pip3 install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip3 install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
cd droid_trajectory/droid_slam
python3 setup.py install
```

Download the outdoor model checkpoints from [here](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth).

Download the droid.pth checkpoint from [here](https://github.com/princeton-vl/DROID-SLAM).


## How to Run

Check `run_trajectory_inference.sbatch`.
