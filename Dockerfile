# PARENT_IMAGE from https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# docker build --tag pytorch . 
# nvidia-docker run --ipc=host --gpus all -it --rm -v "$(pwd):/workspace/amazon" -v "$(pwd)/../my-app/data/model_build_inputs:/workspace/data" pytorch:latest
# python train_classifier.py --datapath ../data --epochs 100

ARG PARENT_IMAGE="nvcr.io/nvidia/pytorch:20.12-py3"
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.8

ENV DEBIAN_FRONTEND="noninteractive" TZ="America/Houston"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code

# Copy setup file only to install dependencies
COPY ./requirements.txt requirements.txt

# Install pip packages
RUN pip install -r requirements.txt && rm requirements.txt

# Download zsh for better terminal
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t robbyrussell && \
    echo "exec zsh" >> ~/.bashrc && \
    conda init zsh && \
    echo 'ZSH_THEME="robbyrussell"' >> ~/.zshrc


CMD /bin/bash