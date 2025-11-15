"""
Image loading and conversion module
Handles ALL image formats (HEIC, PNG, JPG, JPEG, WebP, etc.) while maintaining quality
"""
import os
from pathlib import Path
from PIL import Image
import pillow_heif
from utils.logger import logger

class ImageLoader:
    def __init__(self, config):
        self.config = config
        self.max_dimension = config.get('image_processing', {}).get('max_dimension', 2048)
        
        # Register HEIF/HEIC support
        pillow_heif.register_heif_opener()
        
        # Supported formats
        self.supported_formats = {
            '.heic', '.heif',  # Apple formats
            '.jpg', '.jpeg',    # JPEG
            '.png',             # PNG
            '.webp',            # WebP
            '.bmp',             # Bitmap
            '.tiff', '.tif',    # TIFF
        }
        
        logger.info("ImageLoader initialized with support for all major formats")
    
    def load_image(self, image_path):
        """
        Load image from ANY format
        
        Args:
            image_path: Path to input image
            
        Returns:
            PIL.Image object in RGB mode
        """
        try:
            image_path = Path(image_path)
            
            # Check if file exists
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Check format
            ext = image_path.suffix.lower()
            if ext not in self.supported_formats:
                logger.warning(f"Uncommon format: {ext}. Attempting to load anyway...")
            
            logger.info(f"Loading image: {image_path.name} ({ext})")
            
            # Open image (PIL + pillow_heif handles all formats)
            image = Image.open(image_path)
            
            # Log original mode
            original_mode = image.mode
            logger.debug(f"Original image mode: {original_mode}")
            
            # Convert to RGB (removes alpha channel, handles all modes)
            if image.mode != 'RGB':
                logger.debug(f"Converting from {image.mode} to RGB")
                
                # Special handling for different modes
                if image.mode == 'RGBA':
                    # Create white background for transparency
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])  # Use alpha as mask
                    image = background
                elif image.mode == 'P':
                    # Palette mode
                    image = image.convert('RGB')
                elif image.mode == 'L':
                    # Grayscale
                    image = image.convert('RGB')
                elif image.mode in ['CMYK', 'LAB', 'YCbCr']:
                    image = image.convert('RGB')
                else:
                    # Generic conversion
                    image = image.convert('RGB')
            
            # Log dimensions
            logger.info(f"Loaded image: {image.size[0]}x{image.size[1]} ({original_mode} → RGB)")
            
            # Resize if necessary while maintaining aspect ratio
            image = self._resize_if_needed(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def _resize_if_needed(self, image):
        """Resize image if it exceeds max dimensions while maintaining aspect ratio"""
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > self.max_dimension:
            scale = self.max_dimension / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            logger.info(f"Resizing from {width}x{height} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def save_image(self, image, output_path, quality=95):
        """
        Save image to disk with high quality
        
        Args:
            image: PIL Image object
            output_path: Output file path
            quality: JPEG quality (1-95), ignored for PNG
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format from extension
            ext = output_path.suffix.lower()
            
            if ext in ['.png']:
                image.save(output_path, 'PNG', optimize=True)
            elif ext in ['.jpg', '.jpeg']:
                image.save(output_path, 'JPEG', quality=quality, optimize=True)
            elif ext in ['.webp']:
                image.save(output_path, 'WebP', quality=quality)
            else:
                # Default to PNG for unknown extensions
                output_path = output_path.with_suffix('.png')
                image.save(output_path, 'PNG', optimize=True)
            
            logger.info(f"Image saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {str(e)}")
            raise
    
    def convert_to_png(self, image_path, output_dir=None):
        """
        Convert any image format to PNG
        
        Args:
            image_path: Path to input image
            output_dir: Output directory (defaults to same as input)
            
        Returns:
            Path to converted PNG file
        """
        image_path = Path(image_path)
        
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)
        
        output_path = output_dir / f"{image_path.stem}.png"
        
        # If already PNG, just return the path
        if image_path.suffix.lower() == '.png':
            logger.info("Image already in PNG format")
            return image_path
        
        logger.info(f"Converting {image_path.suffix} to PNG...")
        image = self.load_image(image_path)
        self.save_image(image, output_path)
        
        return output_path
    
    def is_supported_format(self, file_path):
        """Check if file format is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_formats