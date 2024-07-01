python3 setup.py install

# To run the cuda code directly without pybind11:
# nvcc -o gaussian_diffusion main.cu -lcurand
# ./gaussian_diffusion