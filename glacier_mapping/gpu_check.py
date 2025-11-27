import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("❌ CUDA is NOT available.")
    exit(1)

print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Actual GPU test: allocate and compute on GPU
x = torch.rand((5000, 5000), device="cuda")
y = torch.rand((5000, 5000), device="cuda")

print("Running matrix multiply on GPU...")
z = x @ y  # Heavy compute

torch.cuda.synchronize()  # Ensure the GPU finishes before measuring

print("✔ GPU compute succeeded. Result tensor shape:", z.shape)
print("Done.")
