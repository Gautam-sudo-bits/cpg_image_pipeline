"""
Mask processing and refinement module
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from utils.logger import logger

class MaskProcessor:
    def __init__(self, config):
        self.config = config
        self.mask_config = config.get('mask_processing', {})
        logger.info("MaskProcessor initialized")
    
    def process_mask(self, mask, operation='for_inpainting'):
        """
        Process mask for specific operations
        
        Args:
            mask: PIL Image in L mode
            operation: 'for_inpainting' or 'for_compositing'
            
        Returns:
            Processed mask
        """
        try:
            logger.info(f"Processing mask for: {operation}")
            
            if operation == 'for_inpainting':
                # For inpainting, we need inverted mask with feathering
                processed = self._prepare_inpainting_mask(mask)
            elif operation == 'for_compositing':
                # For compositing, we need clean edges with slight blur
                processed = self._prepare_compositing_mask(mask)
            else:
                logger.warning(f"Unknown operation: {operation}, returning original")
                processed = mask
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing mask: {str(e)}")
            raise
    
    def _prepare_inpainting_mask(self, mask):
        """
        Prepare mask for ControlNet inpainting
        Inverted: white=area to inpaint, black=area to preserve
        """
        # Expand mask to ensure product is fully protected
        expanded = self._expand_mask(mask, 
            pixels=self.mask_config.get('expand_mask', 10))
        
        # Feather edges
        feathered = self._feather_edges(expanded,
            pixels=self.mask_config.get('feather_pixels', 3))
        
        # Invert: background becomes white (to be inpainted)
        inverted = ImageOps.invert(feathered)
        
        logger.debug("Inpainting mask prepared")
        return inverted
    
    def _prepare_compositing_mask(self, mask):
        """
        Prepare mask for compositing operations
        Keeps original orientation with smoothing
        """
        # Slight blur for smooth edges
        blur_amount = self.mask_config.get('blur_amount', 5)
        blurred = mask.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        
        logger.debug("Compositing mask prepared")
        return blurred
    
    def _expand_mask(self, mask, pixels=10):
        """
        Expand mask by dilating it
        
        Args:
            mask: PIL Image in L mode
            pixels: Number of pixels to expand
            
        Returns:
            Expanded mask
        """
        try:
            if pixels <= 0:
                return mask
            
            # Convert to numpy for easier manipulation
            mask_array = np.array(mask)
            
            # Simple dilation using maximum filter
            from scipy.ndimage import maximum_filter
            expanded_array = maximum_filter(mask_array, size=pixels*2+1)
            
            expanded = Image.fromarray(expanded_array.astype(np.uint8), mode='L')
            logger.debug(f"Mask expanded by {pixels} pixels")
            return expanded
            
        except ImportError:
            # Fallback if scipy not available - use PIL's MaxFilter
            logger.warning("scipy not available, using PIL MaxFilter")
            expanded = mask.filter(ImageFilter.MaxFilter(size=pixels*2+1))
            return expanded
    
    def _feather_edges(self, mask, pixels=3):
        """
        Feather (soften) mask edges
        
        Args:
            mask: PIL Image in L mode
            pixels: Feather amount
            
        Returns:
            Feathered mask
        """
        if pixels <= 0:
            return mask
        
        # Gaussian blur for smooth feathering
        feathered = mask.filter(ImageFilter.GaussianBlur(radius=pixels))
        logger.debug(f"Mask feathered by {pixels} pixels")
        return feathered
    
    def refine_mask(self, mask, threshold=128):
        """
        Refine mask by applying threshold
        
        Args:
            mask: PIL Image in L mode
            threshold: Threshold value (0-255)
            
        Returns:
            Binary mask
        """
        try:
            mask_array = np.array(mask)
            binary_array = (mask_array > threshold).astype(np.uint8) * 255
            refined = Image.fromarray(binary_array, mode='L')
            logger.debug(f"Mask refined with threshold={threshold}")
            return refined
        except Exception as e:
            logger.error(f"Error refining mask: {str(e)}")
            raise
    
    def visualize_mask_overlay(self, image, mask, color=(255, 0, 0), alpha=0.5):
        """
        Create visualization of mask overlaid on image
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L mode)
            color: RGB tuple for mask color
            alpha: Transparency (0-1)
            
        Returns:
            PIL Image with mask overlay
        """
        try:
            # Create colored mask
            colored_mask = Image.new('RGB', image.size, color)
            
            # Use mask as alpha
            mask_array = np.array(mask).astype(float) / 255.0 * alpha
            
            # Blend
            image_array = np.array(image).astype(float)
            colored_array = np.array(colored_mask).astype(float)
            
            blended_array = (
                image_array * (1 - mask_array[:, :, np.newaxis]) +
                colored_array * mask_array[:, :, np.newaxis]
            ).astype(np.uint8)
            
            blended = Image.fromarray(blended_array, mode='RGB')
            return blended
            
        except Exception as e:
            logger.error(f"Error creating mask overlay: {str(e)}")
            raise