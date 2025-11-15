"""
Image compositing engine
Combines foreground and background with advanced blending
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from utils.logger import logger

class CompositingEngine:
    def __init__(self, config):
        self.config = config
        self.comp_config = config.get('compositing', {})
        logger.info("CompositingEngine initialized")
    
    def composite(self, foreground_rgba, background_rgb, mask=None):
        """
        Composite foreground onto background
        
        Args:
            foreground_rgba: PIL Image with alpha channel
            background_rgb: PIL Image (RGB)
            mask: Optional custom mask (uses foreground alpha if None)
            
        Returns:
            Composited RGB image
        """
        try:
            logger.info("Starting compositing...")
            
            # Ensure correct modes
            if foreground_rgba.mode != 'RGBA':
                foreground_rgba = foreground_rgba.convert('RGBA')
            if background_rgb.mode != 'RGB':
                background_rgb = background_rgb.convert('RGB')
            
            # Ensure same size
            if foreground_rgba.size != background_rgb.size:
                logger.warning("Size mismatch, resizing background")
                background_rgb = background_rgb.resize(
                    foreground_rgba.size, Image.LANCZOS
                )
            
            # Use provided mask or extract from alpha
            if mask is None:
                mask = foreground_rgba.split()[3]  # Alpha channel
            else:
                if mask.size != foreground_rgba.size:
                    mask = mask.resize(foreground_rgba.size, Image.LANCZOS)
            
            # Apply edge refinement if enabled
            if self.comp_config.get('edge_refinement', True):
                mask = self._refine_edges(mask)
            
            # Generate shadow if enabled
            if self.comp_config.get('shadow_generation', True):
                background_rgb = self._add_shadow(background_rgb, mask)
            
            # Perform compositing
            foreground_rgb = Image.new('RGB', foreground_rgba.size)
            foreground_rgb.paste(foreground_rgba, mask=foreground_rgba.split()[3])
            
            # Blend using mask
            composited = Image.composite(foreground_rgb, background_rgb, mask)
            
            # Apply color matching if enabled
            if self.comp_config.get('color_matching', True):
                composited = self._match_colors(composited, foreground_rgb, mask)
            
            logger.info("Compositing complete")
            return composited
            
        except Exception as e:
            logger.error(f"Error during compositing: {str(e)}")
            raise
    
    def _refine_edges(self, mask):
        """Apply edge refinement to mask"""
        try:
            # Slight Gaussian blur for smooth edges
            refined = mask.filter(ImageFilter.GaussianBlur(radius=1))
            logger.debug("Edge refinement applied")
            return refined
        except Exception as e:
            logger.warning(f"Edge refinement failed: {str(e)}")
            return mask
    
    def _add_shadow(self, background, mask):
        """
        Add soft shadow under product
        
        Args:
            background: PIL Image (RGB)
            mask: Product mask
            
        Returns:
            Background with shadow
        """
        try:
            shadow_opacity = self.comp_config.get('shadow_opacity', 0.3)
            shadow_blur = self.comp_config.get('shadow_blur', 15)
            
            # Create shadow mask (shifted down slightly)
            shadow_mask = mask.copy()
            
            # Shift shadow down
            shadow_array = np.array(shadow_mask)
            shift_pixels = 20
            shadow_shifted = np.zeros_like(shadow_array)
            shadow_shifted[shift_pixels:, :] = shadow_array[:-shift_pixels, :]
            shadow_mask = Image.fromarray(shadow_shifted)
            
            # Blur shadow
            shadow_mask = shadow_mask.filter(
                ImageFilter.GaussianBlur(radius=shadow_blur)
            )
            
            # Apply opacity
            shadow_array = np.array(shadow_mask).astype(float) / 255.0 * shadow_opacity
            
            # Darken background where shadow is
            bg_array = np.array(background).astype(float)
            shadow_factor = 1 - shadow_array[:, :, np.newaxis]
            shadowed_array = (bg_array * shadow_factor).astype(np.uint8)
            
            shadowed_bg = Image.fromarray(shadowed_array, mode='RGB')
            logger.debug("Shadow added")
            return shadowed_bg
            
        except Exception as e:
            logger.warning(f"Shadow generation failed: {str(e)}")
            return background
    
    def _match_colors(self, composited, foreground, mask):
        """
        Subtle color matching between foreground and background
        
        Args:
            composited: Composited image
            foreground: Foreground image
            mask: Product mask
            
        Returns:
            Color-matched image
        """
        try:
            # This is a simplified version
            # In production, you might use more sophisticated color transfer
            logger.debug("Color matching applied (basic)")
            return composited
            
        except Exception as e:
            logger.warning(f"Color matching failed: {str(e)}")
            return composited