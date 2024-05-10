#include <torch/extension.h>

// CUDA kernel for element-wise addition
__global__ void add_kernel(float *a, float *b, float *c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// C++ interface to call CUDA kernel
void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int size = a.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Element-wise addition (CUDA)");
}
