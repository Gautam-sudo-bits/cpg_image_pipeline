"""
Model handlers for CPG image generation
"""
try:
    from utils.logger import logger
    
    logger.info("  - Importing NanoBananaAPI...")
    from models.nano_banana_api import NanoBananaAPI
    
    logger.info("  - Importing ControlNetHandler...")
    from models.controlnet_handler import ControlNetHandler
    
    logger.info("  [OK] All models imported")
    
except Exception as e:
    print(f"ERROR in models/__init__.py: {e}")
    import traceback
    traceback.print_exc()
    raise

__all__ = ['NanoBananaAPI', 'ControlNetHandler']