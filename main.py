import torch
import cuda_add

# Create random tensors
a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = torch.empty_like(a)

# Perform addition on GPU
cuda_add.add(a, b, c)

print(c)  # Output of the addition
