"""
ControlNet handler for inpainting using Qwen-Image-ControlNet
Follows official Hugging Face implementation
"""
import torch, os
from PIL import Image
from utils.logger import logger
from utils.gpu_checker import gpu_checker


class ControlNetHandler:
    def __init__(self, config):
        self.config = config
        self.cn_config = config.get('controlnet', {})
        self.device = gpu_checker.get_device()
        
        # Model names
        self.base_model = "Qwen/Qwen-Image"
        self.controlnet_model = self.cn_config.get(
            'model_name', 
            'InstantX/Qwen-Image-ControlNet-Inpainting'
        )
        
        # Pipeline not loaded yet
        self.pipe = None
        self._model_loaded = False
        
        logger.info(f"ControlNetHandler initialized")
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"ControlNet: {self.controlnet_model}")
        logger.info("(Models will load on first use)")
    
    def _load_model(self):
        """Load Qwen-Image ControlNet pipeline with memory optimization"""
        if self._model_loaded:
            return
        
        try:
            logger.info("="*60)
            logger.info("Loading Qwen-Image ControlNet Pipeline")
            logger.info("="*60)
            logger.info("MEMORY-OPTIMIZED MODE")
            
            # Check available memory
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024**3)
                logger.info(f"Available RAM: {available_gb:.2f} GB")
                if available_gb < 8:
                    logger.warning(f"LOW RAM! Only {available_gb:.2f} GB available.")
                    logger.warning("Close other applications!")
            except:
                pass
            
            # Force sequential downloads
            os.environ['HF_HUB_DOWNLOAD_WORKERS'] = '1'
            
            # Use float16 (less memory than bfloat16)
            if self.device.type == "cuda":
                dtype = torch.float16
                logger.info("Using torch.float16 (memory optimized)")
            else:
                dtype = torch.float32
                logger.info("Using torch.float32 for CPU")
            
            # Import classes
            logger.info("[1/3] Importing diffusers classes...")
            from diffusers import QwenImageControlNetModel, QwenImageControlNetInpaintPipeline
            logger.info("  [OK] Classes imported")
            
            # Load ControlNet WITHOUT device_map (not supported)
            logger.info(f"[2/3] Loading ControlNet: {self.controlnet_model}")
            logger.info("Using low_cpu_mem_usage mode...")
            
            controlnet = QwenImageControlNetModel.from_pretrained(
                self.controlnet_model, 
                torch_dtype=dtype,
                low_cpu_mem_usage=True,  # Critical for low RAM
            )
            logger.info("  [OK] ControlNet loaded")
            
            # Clear cache before loading big model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info("  Cleared GPU cache")
            
            # Load pipeline with memory optimization
            logger.info(f"[3/3] Loading pipeline: {self.base_model}")
            logger.info("This may take 5-15 minutes (loading 35GB)...")
            
            self.pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
                self.base_model, 
                controlnet=controlnet, 
                torch_dtype=dtype,
                low_cpu_mem_usage=True,  # Load models one at a time
            )
            logger.info("  [OK] Pipeline loaded")
            
            # Move to device FIRST, then enable optimizations
            logger.info(f"Moving pipeline to {self.device}...")
            self.pipe = self.pipe.to(self.device)
            logger.info("  [OK] Moved to device")
            
            # Enable memory optimizations AFTER moving to device
            logger.info("Enabling memory optimizations...")
            
            if self.device.type == "cuda":
                # Attention slicing (reduces VRAM usage)
                try:
                    self.pipe.enable_attention_slicing(slice_size="max")
                    logger.info("  [OK] Attention slicing")
                except Exception as e:
                    logger.debug(f"  Attention slicing failed: {e}")
                
                # VAE slicing
                try:
                    self.pipe.enable_vae_slicing()
                    logger.info("  [OK] VAE slicing")
                except Exception as e:
                    logger.debug(f"  VAE slicing failed: {e}")
                
                # VAE tiling (for large images)
                try:
                    self.pipe.enable_vae_tiling()
                    logger.info("  [OK] VAE tiling")
                except Exception as e:
                    logger.debug(f"  VAE tiling failed: {e}")
                
                # Clear cache
                torch.cuda.empty_cache()
                logger.info("  [OK] Cleared GPU cache")
            
            self._model_loaded = True
            logger.info("="*60)
            logger.info("[SUCCESS] Qwen-Image ControlNet ready!")
            logger.info("="*60)
            
            # Report memory usage
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024**3)
                logger.info(f"RAM after loading: {available_gb:.2f} GB available")
                if self.device.type == "cuda":
                    gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
                    logger.info(f"GPU VRAM used: {gpu_used:.2f} GB")
            except:
                pass
            
        except Exception as e:
            logger.error(f"[ERROR] Model loading failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def inpaint_background(self, image, mask, prompt, negative_prompt=None):
        """
        Inpaint background - follows official Qwen-Image API
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L mode) - white=inpaint, black=preserve
            prompt: Descriptive prompt
            negative_prompt: Negative prompt (optional)
        
        Returns:
            PIL Image with inpainted background
        """
        
        # Load model if needed
        if not self._model_loaded:
            self._load_model()
        
        try:
            logger.info("Starting Qwen-Image ControlNet inpainting...")
            logger.info(f"Prompt: {prompt[:150]}...")
            
            # Ensure correct formats
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            original_size = image.size
            logger.info(f"Image size: {original_size[0]}x{original_size[1]}")
            
            # Resize to multiples of 64
            target_size = self._get_optimal_size(original_size)
            if target_size != original_size:
                logger.info(f"Resizing to {target_size[0]}x{target_size[1]}")
                control_image = image.resize(target_size, Image.Resampling.LANCZOS)
                control_mask = mask.resize(target_size, Image.Resampling.LANCZOS)
            else:
                control_image = image
                control_mask = mask
            
            # Get parameters
            if negative_prompt is None:
                negative_prompt = self.cn_config.get('negative_prompt', ' ')
            
            num_inference_steps = self.cn_config.get('num_inference_steps', 30)
            true_cfg_scale = self.cn_config.get('true_cfg_scale', 4.0)
            controlnet_conditioning_scale = self.cn_config.get('controlnet_conditioning_scale', 1.0)
            
            # Clear cache
            if self.device.type == "cuda":
                gpu_checker.clear_cache()
            
            # Generate (exactly like official example)
            logger.info(f"Running inference ({num_inference_steps} steps)...")
            logger.info("This may take 1-3 minutes...")
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                control_mask=control_mask,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=control_image.size[0],
                height=control_image.size[1],
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).images[0]
            
            # Resize back if needed
            if result.size != original_size:
                logger.info(f"Resizing back to {original_size[0]}x{original_size[1]}")
                result = result.resize(original_size, Image.Resampling.LANCZOS)
            
            logger.info("[SUCCESS] Inpainting complete!")
            
            # Clear cache
            if self.device.type == "cuda":
                gpu_checker.clear_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Inpainting failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _get_optimal_size(self, original_size, max_dim=1024, multiple=64):
        """Get size that's a multiple of 64"""
        width, height = original_size
        
        if width <= max_dim and height <= max_dim:
            width = (width // multiple) * multiple
            height = (height // multiple) * multiple
            width = max(width, multiple)
            height = max(height, multiple)
            return (width, height)
        
        aspect_ratio = width / height
        if width > height:
            new_width = max_dim
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_dim
            new_width = int(new_height * aspect_ratio)
        
        new_width = (new_width // multiple) * multiple
        new_height = (new_height // multiple) * multiple
        new_width = max(new_width, multiple)
        new_height = max(new_height, multiple)
        
        return (new_width, new_height)
    
    def generate_variations(self, image, mask, prompt, num_variations=3):
        """Generate multiple variations"""
        if not self._model_loaded:
            self._load_model()
        
        variations = []
        for i in range(num_variations):
            logger.info(f"Variation {i+1}/{num_variations}")
            result = self.inpaint_background(image, mask, prompt)
            variations.append(result)
        
        return variations
    """
    def _inpaint_with_seed(self, image, mask, prompt, seed=42):
        
        # Ensure correct formats
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        original_size = image.size
        target_size = self._get_optimal_size(original_size)
        
        if target_size != original_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            mask = mask.resize(target_size, Image.Resampling.LANCZOS)
        
        # Get parameters
        negative_prompt = self.cn_config.get('negative_prompt', ' ')
        num_inference_steps = self.cn_config.get('num_inference_steps', 30)
        true_cfg_scale = self.cn_config.get('true_cfg_scale', 4.0)
        controlnet_conditioning_scale = self.cn_config.get('controlnet_conditioning_scale', 1.0)
        
        # Clear cache
        if self.device.type == "cuda":
            gpu_checker.clear_cache()
        
        # Generate with specific seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=image,
            control_mask=mask,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=image.size[0],
            height=image.size[1],
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        )
        
        result = output.images[0]
        
        # Resize back
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        # Clear cache
        if self.device.type == "cuda":
            gpu_checker.clear_cache()
        
        return result"""