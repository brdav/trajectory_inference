# Base image (torch 2.1.0 and cuda 12.1)
FROM nvcr.io/nvidia/pytorch:23.07-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/home/workspace/trajectory_inference:$PYTHONPATH

# Update essentials
RUN apt-get update && apt-get install python3-pip python3-venv -y
RUN pip install --upgrade pip setuptools

RUN apt-get install -y git && \
    apt-get install -y software-properties-common && \
    apt-get install -y ninja-build && \
    apt-get install -y build-essential && \
    apt-get install -y cmake

# Create a working directory
RUN mkdir -p /home/workspace
WORKDIR /home/workspace

# Clone the trajectory_inference repository recursively
RUN git clone --recursive git@github.com:brdav/trajectory_inference.git

# Install Python dependencies
RUN pip3 install numpy opencv-python h5py scipy tensorboard && \
    pip3 install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html && \
    pip3 install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu121

# Compile DROID-SLAM CUDA extension
RUN cd /home/workspace/trajectory_inference/droid_trajectory/droid_slam && \
    python3 setup.py install

RUN cd /home/workspace/trajectory_inference && \
    pip3 install -e .

# Set up aliases for convenience
RUN echo 'alias python=python3' >> ~/.bashrc && \
    echo 'alias pip=pip3' >> ~/.bashrc

# Entry point
CMD ["/bin/bash"]