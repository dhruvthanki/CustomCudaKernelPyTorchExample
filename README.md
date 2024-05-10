Install a specific version of nvidia driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-XXX

Use the following link to setup the Cuda Toolkit
https://developer.nvidia.com/cuda-downloads

Add these lines at the end of the bash file:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

verify the installation
nvcc --version

Compile and bind the Cuda code:
python3 setup.py install

Run the python code that depends on the custom cuda kernel:
python3 main.py