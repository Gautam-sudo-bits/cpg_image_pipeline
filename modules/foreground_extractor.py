"""
Foreground extraction module using rembg
Extracts product from background and creates masks
"""
import numpy as np
from PIL import Image
from rembg import remove, new_session
from utils.logger import logger

class ForegroundExtractor:
    def __init__(self, config):
        self.config = config
        self.fg_config = config.get('foreground_extraction', {})
        self.model_name = self.fg_config.get('model', 'u2net')
        
        # Don't initialize session yet - lazy loading
        self.session = None
        self._session_initialized = False
        
        logger.info(f"ForegroundExtractor initialized (model: {self.model_name}, will load on first use)")
    
    def _initialize_session(self):
        """Initialize rembg session (lazy loading)"""
        if self._session_initialized:
            return
        
        try:
            logger.info("="*60)
            logger.info("Initializing rembg (Background Removal)")
            logger.info("="*60)
            logger.info(f"Model: {self.model_name}")
            logger.info("Downloading model on first run (this may take a few minutes)...")
            logger.info("Model will be cached at: ~/.u2net/")
            
            # Create rembg session
            self.session = new_session(self.model_name)
            
            self._session_initialized = True
            logger.info("="*60)
            logger.info("rembg initialized successfully!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error initializing rembg: {str(e)}")
            raise
    
    def extract_foreground(self, image):
        """
        Extract foreground from image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing:
                - foreground: RGBA image with transparent background
                - mask: Binary mask (L mode)
                - original: Original RGB image
        """
        
        # Initialize session if not already done
        if not self._session_initialized:
            self._initialize_session()
        
        try:
            logger.info("Extracting foreground from image...")
            
            # Convert PIL Image to bytes for rembg
            original_rgb = image.convert('RGB')
            
            # Extract foreground with alpha matting if enabled
            if self.fg_config.get('alpha_matting', True):
                logger.debug("Using alpha matting for better edge quality")
                foreground_rgba = remove(
                    original_rgb,
                    session=self.session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=self.fg_config.get(
                        'alpha_matting_foreground_threshold', 240
                    ),
                    alpha_matting_background_threshold=self.fg_config.get(
                        'alpha_matting_background_threshold', 10
                    ),
                    alpha_matting_erode_size=self.fg_config.get(
                        'alpha_matting_erode_size', 10
                    )
                )
            else:
                foreground_rgba = remove(original_rgb, session=self.session)
            
            # Extract alpha channel as mask
            if isinstance(foreground_rgba, np.ndarray):
                foreground_rgba = Image.fromarray(foreground_rgba)
            
            # Ensure RGBA mode
            if foreground_rgba.mode != 'RGBA':
                foreground_rgba = foreground_rgba.convert('RGBA')
            
            # Extract mask from alpha channel
            mask = foreground_rgba.split()[3]  # Get alpha channel
            
            logger.info("Foreground extraction complete")
            
            return {
                'foreground': foreground_rgba,
                'mask': mask,
                'original': original_rgb
            }
            
        except Exception as e:
            logger.error(f"Error extracting foreground: {str(e)}")
            raise
    
    def get_bounding_box(self, mask):
        """
        Get bounding box of the foreground object
        
        Args:
            mask: PIL Image in L mode
            
        Returns:
            Tuple (x, y, width, height)
        """
        try:
            bbox = mask.getbbox()  # Returns (left, upper, right, lower)
            if bbox:
                x, y, right, lower = bbox
                width = right - x
                height = lower - y
                logger.debug(f"Bounding box: x={x}, y={y}, w={width}, h={height}")
                return (x, y, width, height)
            else:
                logger.warning("No foreground detected in mask")
                return None
        except Exception as e:
            logger.error(f"Error getting bounding box: {str(e)}")
            raise
    
    def create_inverted_mask(self, mask):
        """
        Create inverted mask (for background inpainting)
        
        Args:
            mask: PIL Image in L mode (white=foreground, black=background)
            
        Returns:
            Inverted mask (white=background, black=foreground)
        """
        try:
            from PIL import ImageOps
            inverted = ImageOps.invert(mask)
            logger.debug("Inverted mask created")
            return inverted
        except Exception as e:
            logger.error(f"Error creating inverted mask: {str(e)}")
            raise