# Ultralytics YOLO 🚀, AGPL-3.0 license
# Builds ultralytics/ultralytics:latest image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CUDA-optimized for YOLOv8 single/multi-GPU training and inference

# Variables used at build time.
## Base CUDA version. See all supported version at https://hub.docker.com/r/nvidia/cuda/tags?page=2&name=-devel-ubuntu
ARG CUDA_VERSION=11.8
## Base Ubuntu version.
ARG TORCH_VERSION=2.3.1

# Define base image.
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn8-devel AS base

# Duplicate args because of the visibility zone
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CUDA_VERSION
ARG TORCH_VERSION
## Base Timezone
ARG TZ=Europe/Moscow

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive \
    ## Set timezone as it is required by some packages.
    TZ=${TZ} \
    ## Accelerate compilation flags (use all cores)
    MAKEFLAGS=-j$(nproc) \
    ## Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library" https://github.com/pytorch/pytorch/issues/37377
    MKL_THREADING_LAYER=GNU \
    # Set environment variables
    APP_HOME=/usr/src/drone_hack

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
# libsm6 required by libqxcb to create QT-based windows for visualization; set 'QT_DEBUG_PLUGINS=1' to test in docker
RUN apt update \
    && apt install \
        --no-install-recommends \
        --yes \
            gcc \
            git \
            zip \
            curl \
            libgl1 \
            libglib2.0-0 \
            libpython3-dev \
            gnupg \
            g++ \
            libusb-1.0-0 \
            libsm6

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade \
    --no-install-recommends \
    --yes \
        openssl \
        tar

# Create working directory
WORKDIR $APP_HOME

# Copy contents and assign permissions
COPY . $APP_HOME

# Install pip packages
RUN python3 -m pip install \
    --no-cache-dir \
    --upgrade \
        pip \
        wheel && \
    python3 -m pip install \
    --requirement \
        requirements.txt && \
    python3 -m pip install -e \
        my_classificationlib && \
    python3 -m pip uninstall \
    --yes \
        opencv-python \
        opencv-contrib-python && \
    conda update \
    --yes \
        conda && \
    conda install \
    --yes -c conda-forge \
        opencv && \
    python3 -m pip install -e \
        sahi

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/ultralytics:latest && sudo docker build -f docker/Dockerfile -t $t . && sudo docker push $t

# Pull and Run with access to all GPUs
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with access to GPUs 2 and 3 (inside container CUDA devices will appear as 0 and 1)
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus '"device=2,3"' $t

# Pull and Run with local directory access
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/ultralytics:latest)

# DockerHub tag update
# t=ultralytics/ultralytics:latest tnew=ultralytics/ultralytics:v6.2 && sudo docker pull $t && sudo docker tag $t $tnew && sudo docker push $tnew

# Clean up
# sudo docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/ultralytics:latest