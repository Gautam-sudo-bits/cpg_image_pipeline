# test_memory.py
import torch
import psutil

print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

if torch.cuda.is_available():
    print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"GPU Free VRAM: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")
else:
    print("No GPU detected")