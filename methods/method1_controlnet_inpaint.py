"""
Method 1: Pure ControlNet Inpainting
Uses ControlNet to inpaint background while preserving product
Uses DESCRIPTIVE prompts (not instructional) describing entire scene
"""
from pathlib import Path
from modules import (
    ImageLoader, ForegroundExtractor, MaskProcessor, 
    CompositingEngine, Visualizer
)
from models import ControlNetHandler
from utils import logger
from utils.prompt_builder import SmartPromptEnhancer
from utils.auto_prompt_generator import AutoPromptGenerator

class Method1ControlNetInpaint:
    def __init__(self, config, api_key=None):
        self.config = config
        
        logger.info("="*60)
        logger.info("Initializing Method 1: Pure ControlNet Inpainting")
        logger.info("="*60)
        
        # Initialize modules
        self.image_loader = ImageLoader(config)
        self.fg_extractor = ForegroundExtractor(config)
        self.mask_processor = MaskProcessor(config)
        self.visualizer = Visualizer(config)
        
        # Initialize ControlNet
        self.controlnet = ControlNetHandler(config)
        
        # Initialize Auto Prompt Generator
        self.auto_prompt_gen = AutoPromptGenerator(api_key)
        
        # Initialize Smart Prompt Enhancer if enabled
        self.prompt_enhancer = None
        if config.get('prompt_enhancement', {}).get('use_smart_enhancer', True):
            try:
                self.prompt_enhancer = SmartPromptEnhancer(api_key)
                logger.info("Smart Prompt Enhancer enabled")
            except Exception as e:
                logger.warning(f"Could not initialize prompt enhancer: {str(e)}")
        
        logger.info("Method 1 initialized successfully")
    
    def generate(self, input_image_path, prompt=None, product_description=None,
                 style_preset=None, output_path=None):
        """
        Generate CPG image using ControlNet inpainting
        
        Args:
            input_image_path: Path to input product image
            prompt: Background description (OPTIONAL - will auto-generate if None)
            product_description: Description of product (OPTIONAL - will auto-detect if None)
            style_preset: Optional style preset name
            output_path: Optional output path
            
        Returns:
            Path to generated image
        """
        try:
            logger.info(f"Starting Method 1 generation for: {input_image_path}")
            
            # Stage 1: Load image
            logger.info("[Stage 1/6] Loading image...")
            original_image = self.image_loader.load_image(input_image_path)
            
            # Stage 2: Extract foreground
            logger.info("[Stage 2/6] Extracting foreground...")
            extraction_result = self.fg_extractor.extract_foreground(original_image)
            foreground_rgba = extraction_result['foreground']
            mask = extraction_result['mask']
            
            # Stage 3: Process mask for inpainting
            logger.info("[Stage 3/6] Processing mask...")
            inpaint_mask = self.mask_processor.process_mask(
                mask, 
                operation='for_inpainting'
            )
            
            # Stage 4: Auto-generate prompt if not provided
            if prompt is None or product_description is None:
                logger.info("[Stage 4/6] Auto-generating creative prompt...")
                auto_prompt, analysis = self.auto_prompt_gen.generate_for_method1(original_image)
                
                if prompt is None:
                    logger.info("No prompt provided - using auto-generated prompt")
                    final_prompt = auto_prompt
                else:
                    # User provided prompt, but we still need product description
                    if product_description is None:
                        product_description = analysis['product_description']
                    final_prompt = self._build_controlnet_prompt(
                        prompt=prompt,
                        product_description=product_description,
                        style_preset=style_preset,
                        input_path=input_image_path
                    )
            else:
                # Both prompt and product description provided
                logger.info("[Stage 4/6] Building prompt from user input...")
                final_prompt = self._build_controlnet_prompt(
                    prompt=prompt,
                    product_description=product_description,
                    style_preset=style_preset,
                    input_path=input_image_path
                )
            
            logger.info(f"Final prompt: {final_prompt}")
            
            # Stage 5: Inpaint background with ControlNet
            logger.info("[Stage 5/6] Inpainting background with ControlNet...")
            result_image = self.controlnet.inpaint_background(
                image=original_image,
                mask=inpaint_mask,
                prompt=final_prompt
            )
            
            # Stage 6: Save result
            logger.info("[Stage 6/6] Saving result...")
            if output_path is None:
                output_dir = Path(self.config['paths']['output_dir'])
                output_dir.mkdir(exist_ok=True)
                input_name = Path(input_image_path).stem
                output_path = output_dir / f"{input_name}_method1_result.png"
            
            saved_path = self.image_loader.save_image(result_image, output_path)
            
            # Visualize stages
            if self.config.get('visualization', {}).get('show_stages', True):
                logger.info("Creating visualization...")
                self._visualize_pipeline(
                    original_image,
                    foreground_rgba,
                    mask,
                    inpaint_mask,
                    result_image,
                    Path(input_image_path).stem
                )
            
            logger.info(f"✓ Method 1 complete! Output: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error in Method 1 generation: {str(e)}")
            raise
    
    def _build_controlnet_prompt(self, prompt, product_description=None, 
                                style_preset=None, input_path=None):
        """
        Build descriptive prompt for ControlNet (Qwen-Image based)
        
        Format: Describes ENTIRE scene (product + background) in descriptive language
        """
        
        # If product_description not provided, try to infer from filename
        if product_description is None:
            if input_path:
                # Extract product name from filename
                product_name = Path(input_path).stem.replace('_', ' ').replace('-', ' ')
                product_description = f"a {product_name}"
            else:
                product_description = "the product"
        
        # Apply style preset if provided
        if style_preset:
            background_desc = self._apply_style_preset(prompt, style_preset)
        else:
            background_desc = prompt
        
        # Use Smart Prompt Enhancer if available
        if self.prompt_enhancer:
            try:
                logger.info("Enhancing prompt with Gemini 2.0 Pro...")
                enhanced_prompt = self.prompt_enhancer.enhance_for_controlnet(
                    product_name=product_description,
                    background_idea=background_desc,
                    target_audience=self.config.get('product_defaults', {}).get(
                        'target_audience', 'Gen-Z'
                    )
                )
                return enhanced_prompt
            except Exception as e:
                logger.warning(f"Prompt enhancement failed, using fallback: {str(e)}")
        
        # Fallback: Build basic descriptive prompt
        style_desc = self.config.get('product_defaults', {}).get(
            'photography_style', 
            'clean, professional, and visually appealing'
        )
        
        fallback_prompt = (
            f"This photo showcases {product_description}. "
            f"The product is clearly highlighted and prominently displayed, "
            f"placed in {background_desc}. "
            f"The overall image maintains a {style_desc} visual style, "
            f"with natural lighting and high-quality commercial photography aesthetics."
        )
        
        return fallback_prompt
    
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
    
    def _visualize_pipeline(self, original, foreground, mask, inpaint_mask, result, base_name):
        """Visualize all pipeline stages"""
        try:
            # Convert foreground RGBA to RGB for visualization
            fg_rgb = foreground.convert('RGB')
            
            stages = {
                '1. Original': original,
                '2. Extracted Foreground': fg_rgb,
                '3. Foreground Mask': mask.convert('RGB'),
                '4. Inpainting Mask': inpaint_mask.convert('RGB'),
                '5. Final Result': result
            }
            
            self.visualizer.visualize_stages(stages, f"{base_name}_method1_pipeline")
            
            # Also create before/after comparison
            self.visualizer.create_comparison(original, result, f"{base_name}_method1_comparison")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {str(e)}")
    
    def generate_with_variations(self, input_image_path, prompt=None, 
                                product_description=None, num_variations=3):
        """
        Generate multiple variations with different ControlNet parameters
        
        Args:
            input_image_path: Path to input product image
            prompt: Background description (optional)
            product_description: Product description (optional)
            num_variations: Number of variations
            
        Returns:
            List of paths to generated images
        """
        try:
            logger.info(f"Generating {num_variations} variations with Method 1...")
            
            results = []
            input_name = Path(input_image_path).stem
            output_dir = Path(self.config['paths']['output_dir'])
            output_dir.mkdir(exist_ok=True)
            
            for i in range(num_variations):
                logger.info(f"\n--- Variation {i+1}/{num_variations} ---")
                output_path = output_dir / f"{input_name}_method1_var{i+1}.png"
                
                result_path = self.generate(
                    input_image_path=input_image_path,
                    prompt=prompt,
                    product_description=product_description,
                    output_path=output_path
                )
                results.append(result_path)
            
            logger.info(f"✓ Generated {len(results)} variations")
            return results
            
        except Exception as e:
            logger.error(f"Error generating variations: {str(e)}")
            raise