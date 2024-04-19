import torch
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
print(torch.cuda.get_device_name(0))