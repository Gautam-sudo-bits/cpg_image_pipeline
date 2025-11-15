# Create a file: download_models.py

import os
os.environ['HF_HUB_DOWNLOAD_WORKERS'] = '1'

from huggingface_hub import snapshot_download
import torch

print("Downloading ControlNet...")
snapshot_download(
    repo_id="InstantX/Qwen-Image-ControlNet-Inpainting",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("\nDownloading Qwen-Image base model...")
snapshot_download(
    repo_id="Qwen/Qwen-Image",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("\n✅ All models downloaded!")