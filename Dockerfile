FROM nvcr.io/nvidia/pytorch:24.07-py3

# COPY /capstor/scratch/cscs/pmartell/trajectory_inference /app/trajectory_inference
WORKDIR /app/trajectory_inference

RUN nvcc --version
ENV DEBIAN_FRONTEND=noninteractive
# setup
RUN apt-get update && apt-get install python3-pip python3-venv -y
RUN pip install --upgrade pip setuptools
RUN MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0" pip install -v -U git+https://github.com/facebookresearch/xformers.git@2bcbc55#egg=xformers
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y software-properties-common && \
    apt-get install -y ninja-build && \
    apt-get install -y build-essential && \
    apt-get install -y cmake
RUN apt-get install -y openslide-tools
RUN apt-get install -y libpixman-1-0 build-essential cmake ninja-build ffmpeg
RUN pip3 install numpy opencv-python h5py scipy tensorboard p_tqdm
RUN pip3 install torch_scatter
RUN apt-get install libeigen3-dev -y
RUN cd /app/trajectory_inference/droid_trajectory/droid_slam && \
    python3 setup.py install
    
RUN cd /app/trajectory_inference && \
    pip3 install -e .

# Set up aliases for convenience
RUN echo 'alias python=python3' >> ~/.bashrc && \
echo 'alias pip=pip3' >> ~/.bashrc

# Entry point
CMD ["/bin/bash"]