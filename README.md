# Project Setup Guide

## Supported Platforms
- Ubuntu

## Prerequisites
- NVIDIA GPU with appropriate drivers
- Python 3.x

## Installation Steps

### 1. Install NVIDIA Driver
To install a specific version of the NVIDIA driver, execute the following commands:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-XXX

// Install nvidia cuda toolkit
Ref: https://developer.nvidia.com/cuda-downloads

// Add these lines at the end of the bash file:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

// verify the installation
nvcc --version

// Compile and bind the Cuda code:
python3 setup.py install

// Run the python code that depends on the custom cuda kernel:
python3 main.py

```