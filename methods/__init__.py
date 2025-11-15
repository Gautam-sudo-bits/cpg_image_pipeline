"""
Generation methods for CPG images
"""
import sys
sys.path.insert(0, '.')

# Import with detailed logging
try:
    from utils.logger import logger
    logger.info("Importing Method1ControlNetInpaint...")
    from methods.method1_controlnet_inpaint import Method1ControlNetInpaint
    logger.info("[OK] Method1ControlNetInpaint imported")
    
    logger.info("Importing Method2NanoBananaComposite...")
    from methods.method2_nanobanana_composite import Method2NanoBananaComposite
    logger.info("[OK] Method2NanoBananaComposite imported")
    
except Exception as e:
    print(f"ERROR in methods/__init__.py: {e}")
    import traceback
    traceback.print_exc()
    raise

__all__ = ['Method1ControlNetInpaint', 'Method2NanoBananaComposite']