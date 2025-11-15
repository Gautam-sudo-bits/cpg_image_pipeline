"""
Core processing modules for CPG image generation
"""
try:
    from utils.logger import logger
    
    logger.info("  - Importing ImageLoader...")
    from modules.image_loader import ImageLoader
    
    logger.info("  - Importing ForegroundExtractor...")
    from modules.foreground_extractor import ForegroundExtractor
    
    logger.info("  - Importing MaskProcessor...")
    from modules.mask_processor import MaskProcessor
    
    logger.info("  - Importing CompositingEngine...")
    from modules.compositing_engine import CompositingEngine
    
    logger.info("  - Importing Visualizer...")
    from modules.visualizer import Visualizer
    
    logger.info("  [OK] All modules imported")
    
except Exception as e:
    print(f"ERROR in modules/__init__.py: {e}")
    import traceback
    traceback.print_exc()
    raise

__all__ = [
    'ImageLoader',
    'ForegroundExtractor',
    'MaskProcessor',
    'CompositingEngine',
    'Visualizer'
]