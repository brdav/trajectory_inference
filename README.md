# Trajectory Inference

## Setup

Clone the repository:
```bash
git clone --recursive git@github.com:brdav/trajectory_inference.git
```

In an environment with Python>=3.9 and CUDA 12.1:
```bash
pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install opencv-python h5py scipy tensorboard
pip3 install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip3 install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
cd droid_trajectory/droid_slam
python3 setup.py install
```

Download the outdoor model checkpoints from [here](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth).

Download the droid.pth checkpoint from [here](https://github.com/princeton-vl/DROID-SLAM).


## Setup on Todi

[In progress...]

Request an interactive job:
```
ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn srun -A a03 --reservation=sai-a03 --time=1:00:00 --pty bash
```

Navigate to the project directory and build the container:
```bash
cd ~/trajectory_inference
podman build -t trajectory-inference .
enroot import -x mount -o trajectory-inference.sqsh podman://trajectory-inference
```

Create a `.toml` file with instructions:
```bash
mkdir -p ~/.edf
vim ~/.edf/trajectory-inference.toml
```

Adapt the content:
```vim
image = "/your/path/to/trajectory-inference.sqsh"

mounts = ["/capstor", "/users"]

writable = true

workdir = "/iopsstor/scratch/cscs/fozdemir"

[env]
ENROOT_LIBRARY_PATH="/capstor/scratch/cscs/fmohamed/enrootlibn"
LD_LIBRARY_PATH="/usr/local/cuda/lib64:/opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib.real:/external/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

[annotations.hooks.aws_ofi_nccl]
enabled = "true"
variant = "cuda12"

[annotations.com.hooks.ssh]
enabled = "true"
authorize_ssh_key = "/users/dbrggema/.ssh/cscs-key.pub"
```


## How to Run

Check parameters in the `.sbatch` script, then submit with:

```
NODE_IDX="0"
sbatch --export=NODE_IDX=$NODE_IDX run_trajectory_inference.sbatch
```
