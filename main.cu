#include <curand_kernel.h>
#include <stdio.h>

__global__ void gaussian_diffusion_kernel(float *data, int n, float dt, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Initialize the random state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate Gaussian noise
        float noise = curand_normal(&state);

        // Apply the diffusion process
        data[idx] += sqrtf(dt) * noise;
    }
}

void gaussian_diffusion(float *data, int n, float dt) {
    // Allocate device memory
    float *d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, n * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return;
    }

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Call the kernel
    unsigned long long seed = 1234ULL;
    gaussian_diffusion_kernel<<<numBlocks, blockSize>>>(d_data, n, dt, seed);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return;
    }

    err = cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to host failed: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_data);
}

int main() {
    int n = 1024;
    float dt = 0.01f;
    float *data = (float*)malloc(n * sizeof(float));

    // Initialize data
    for (int i = 0; i < n; ++i) {
        data[i] = 0.0f;
    }

    // Perform Gaussian diffusion
    gaussian_diffusion(data, n, dt);

    // Print the first 10 elements
    for (int i = 0; i < 10; ++i) {
        printf("%f\n", data[i]);
    }

    // Free host memory
    free(data);

    return 0;
}
