"""
Method 2: Nano Banana + Composite
Generates background separately with Nano Banana (Gemini 2.0 Flash), then composites product on top
"""
from pathlib import Path
from modules import (
    ImageLoader, ForegroundExtractor, MaskProcessor, 
    CompositingEngine, Visualizer
)
from models import NanoBananaAPI
from utils import logger
from utils.prompt_builder import SmartPromptEnhancer
from utils.auto_prompt_generator import AutoPromptGenerator

class Method2NanoBananaComposite:
    def __init__(self, config, api_key=None):
        self.config = config
        
        logger.info("="*60)
        logger.info("Initializing Method 2: Nano Banana + Composite")
        logger.info("="*60)
        
        # Initialize modules
        self.image_loader = ImageLoader(config)
        self.fg_extractor = ForegroundExtractor(config)
        self.mask_processor = MaskProcessor(config)
        self.compositor = CompositingEngine(config)
        self.visualizer = Visualizer(config)
        
        # Initialize Nano Banana API
        self.nano_banana = NanoBananaAPI(config, api_key)
        
        # Initialize Auto Prompt Generator
        self.auto_prompt_gen = AutoPromptGenerator(api_key)
        
        # Initialize Smart Prompt Enhancer if enabled
        self.prompt_enhancer = None
        if config.get('nano_banana', {}).get('enhance_background_prompts', True):
            try:
                self.prompt_enhancer = SmartPromptEnhancer(api_key)
                logger.info("Smart Prompt Enhancer enabled for backgrounds")
            except Exception as e:
                logger.warning(f"Could not initialize prompt enhancer: {str(e)}")
        
        logger.info("Method 2 initialized successfully")
    
    def generate(self, input_image_path, prompt=None, style_preset=None, 
                 color_palette=None, output_path=None):
        """
        Generate CPG image using Nano Banana background + compositing
        
        Args:
            input_image_path: Path to input product image
            prompt: Background prompt (OPTIONAL - will auto-generate if None)
            style_preset: Optional style preset name
            color_palette: Optional color palette for background
            output_path: Optional output path
            
        Returns:
            Path to generated image
        """
        try:
            logger.info(f"Starting Method 2 generation for: {input_image_path}")
            
            # Stage 1: Load image
            logger.info("[Stage 1/6] Loading image...")
            original_image = self.image_loader.load_image(input_image_path)
            width, height = original_image.size
            
            # Stage 2: Extract foreground
            logger.info("[Stage 2/6] Extracting foreground...")
            extraction_result = self.fg_extractor.extract_foreground(original_image)
            foreground_rgba = extraction_result['foreground']
            mask = extraction_result['mask']
            
            # Stage 3: Process mask for compositing
            logger.info("[Stage 3/6] Processing mask...")
            composite_mask = self.mask_processor.process_mask(
                mask, 
                operation='for_compositing'
            )
            
            # Stage 4: Auto-generate prompt if not provided
            if prompt is None:
                logger.info("[Stage 4/6] Auto-generating creative background prompt...")
                background_prompt, analysis = self.auto_prompt_gen.generate_for_method2(original_image)
                logger.info("No prompt provided - using auto-generated background prompt")
            else:
                logger.info("[Stage 4/6] Building background prompt from user input...")
                background_prompt = self._build_background_prompt(
                    prompt=prompt,
                    style_preset=style_preset,
                    color_palette=color_palette
                )
            
            logger.info(f"Background prompt: {background_prompt}")
            
            # Stage 5: Generate background with Nano Banana
            logger.info("[Stage 5/6] Generating background with Nano Banana (Gemini 2.0 Flash)...")
            
            background_image = self.nano_banana.generate_background(
                prompt=background_prompt,
                reference_image=None,  # Don't pass product image to avoid it appearing in background
                width=width,
                height=height
            )
            
            # Stage 6: Composite foreground onto generated background
            logger.info("[Stage 6/6] Compositing foreground onto background...")
            result_image = self.compositor.composite(
                foreground_rgba=foreground_rgba,
                background_rgb=background_image,
                mask=composite_mask
            )
            
            # Save result
            if output_path is None:
                output_dir = Path(self.config['paths']['output_dir'])
                output_dir.mkdir(exist_ok=True)
                input_name = Path(input_image_path).stem
                output_path = output_dir / f"{input_name}_method2_result.png"
            
            saved_path = self.image_loader.save_image(result_image, output_path)
            
            # Visualize stages
            if self.config.get('visualization', {}).get('show_stages', True):
                logger.info("Creating visualization...")
                self._visualize_pipeline(
                    original_image,
                    foreground_rgba,
                    mask,
                    background_image,
                    result_image,
                    Path(input_image_path).stem
                )
            
            logger.info(f"✓ Method 2 complete! Output: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error in Method 2 generation: {str(e)}")
            raise
    
    def _build_background_prompt(self, prompt, style_preset=None, color_palette=None):
        """
        Build background-only prompt for Nano Banana
        
        Important: Must NOT include product, text, or foreground objects
        """
        
        # Apply style preset if provided
        if style_preset:
            prompt = self._apply_style_preset(prompt, style_preset)
        
        # Get style keywords from config
        style_keywords = self.config.get('nano_banana', {}).get('style_keywords', [])
        
        # Use Smart Prompt Enhancer if available
        if self.prompt_enhancer:
            try:
                logger.info("Enhancing background prompt with Gemini 2.0 Pro...")
                enhanced_prompt = self.prompt_enhancer.enhance_for_nanobanan(
                    background_concept=prompt,
                    style_preferences=style_keywords,
                    color_palette=color_palette
                )
                
                # Ensure safety: explicitly state no products
                enhanced_prompt += " Background only, no products, no text, no foreground objects."
                
                return enhanced_prompt
            except Exception as e:
                logger.warning(f"Prompt enhancement failed, using fallback: {str(e)}")
        
        # Fallback: Build basic background prompt
        style_str = ", ".join(style_keywords[:3]) if style_keywords else "vibrant, creative"
        
        fallback_prompt = (
            f"A creative and visually stunning background scene featuring {prompt}. "
            f"Style: {style_str}. "
            f"High quality commercial photography background with perfect lighting and composition. "
            f"Background only, no products, no text, no objects in foreground."
        )
        
        if color_palette:
            color_str = ", ".join(color_palette)
            fallback_prompt += f" Color palette: {color_str}."
        
        return fallback_prompt
    
    def generate_multiple_variations(self, input_image_path, prompts_list=None, 
                                     style_preset=None, color_palette=None):
        """
        Generate multiple variations with different prompts
        
        Args:
            input_image_path: Path to input product image
            prompts_list: List of background prompts (optional - will auto-generate if None)
            style_preset: Optional style preset
            color_palette: Optional color palette
            
        Returns:
            List of paths to generated images
        """
        try:
            # If no prompts provided, generate variations automatically
            if prompts_list is None:
                num_variations = 3
                logger.info(f"Generating {num_variations} variations with auto-generated prompts...")
                prompts_list = [None] * num_variations  # Will trigger auto-generation for each
            else:
                logger.info(f"Generating {len(prompts_list)} variations...")
            
            results = []
            input_name = Path(input_image_path).stem
            output_dir = Path(self.config['paths']['output_dir'])
            output_dir.mkdir(exist_ok=True)
            
            for idx, prompt in enumerate(prompts_list):
                logger.info(f"\n--- Variation {idx+1}/{len(prompts_list)} ---")
                output_path = output_dir / f"{input_name}_method2_var{idx+1}.png"
                
                result_path = self.generate(
                    input_image_path=input_image_path,
                    prompt=prompt,
                    style_preset=style_preset,
                    color_palette=color_palette,
                    output_path=output_path
                )
                results.append(result_path)
            
            logger.info(f"✓ Generated {len(results)} variations")
            return results
            
        except Exception as e:
            logger.error(f"Error generating variations: {str(e)}")
            raise
    
    def _apply_style_preset(self, base_prompt, preset_name):
        """Apply style preset to prompt"""
        try:
            import yaml
            preset_file = Path('config/style_presets.yaml')
            
            if preset_file.exists():
                with open(preset_file, 'r') as f:
                    presets = yaml.safe_load(f)
                
                if preset_name in presets.get('presets', {}):
                    preset = presets['presets'][preset_name]
                    enhanced_prompt = f"{base_prompt}, {preset['prompt']}"
                    logger.info(f"Applied style preset: {preset_name}")
                    return enhanced_prompt
            
            logger.warning(f"Style preset '{preset_name}' not found, using base prompt")
            return base_prompt
            
        except Exception as e:
            logger.warning(f"Error applying style preset: {str(e)}")
            return base_prompt
    
    def _visualize_pipeline(self, original, foreground, mask, background, result, base_name):
        """Visualize all pipeline stages"""
        try:
            # Convert foreground RGBA to RGB for visualization
            fg_rgb = foreground.convert('RGB')
            
            stages = {
                '1. Original': original,
                '2. Extracted Foreground': fg_rgb,
                '3. Foreground Mask': mask.convert('RGB'),
                '4. Generated Background (Nano Banana)': background,
                '5. Final Composite': result
            }
            
            self.visualizer.visualize_stages(stages, f"{base_name}_method2_pipeline")
            
            # Also create before/after comparison
            self.visualizer.create_comparison(original, result, f"{base_name}_method2_comparison")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {str(e)}")