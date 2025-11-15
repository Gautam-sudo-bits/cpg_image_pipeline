"""
GPU availability checker with automatic CPU fallback
"""
import torch
from utils.logger import logger

class GPUChecker:
    def __init__(self):
        self.device = self._get_device()
        self.device_name = self._get_device_name()
        logger.info(f"Using device: {self.device_name}")
    
    def _get_device(self):
        """Determine the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # For Mac M1/M2
            return torch.device("mps")
        else:
            logger.warning("No GPU detected. Falling back to CPU.")
            return torch.device("cpu")
    
    def _get_device_name(self):
        """Get human-readable device name"""
        if self.device.type == "cuda":
            return f"CUDA GPU: {torch.cuda.get_device_name(0)}"
        elif self.device.type == "mps":
            return "Apple Silicon GPU (MPS)"
        else:
            return "CPU"
    
    def get_device(self):
        return self.device
    
    def get_memory_info(self):
        """Get GPU memory info if available"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2)
            }
        return None
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

# Global GPU checker instance
gpu_checker = GPUChecker()