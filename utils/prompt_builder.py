"""
Prompt builder for generating descriptive prompts
"""
from utils.logger import logger

class PromptBuilder:
    """Build descriptive prompts for different methods"""
    
    def __init__(self, config):
        self.config = config
    
    def build_controlnet_prompt(self, product_description, background_description, 
                               style_description=None):
        """
        Build descriptive prompt for ControlNet (Qwen-Image based)
        
        Format: Describes ENTIRE image (product + background) in descriptive language
        NOT instructional.
        
        Args:
            product_description: Description of the product
            background_description: Desired background
            style_description: Overall style
            
        Returns:
            Formatted descriptive prompt
        """
        
        if style_description is None:
            style_description = "clean, professional, and visually appealing"
        
        prompt = (
            f"This photo showcases {product_description}. "
            f"The product is clearly highlighted and prominently displayed, "
            f"placed in {background_description}. "
            f"The overall image maintains a {style_description} visual style, "
            f"with natural lighting and high-quality commercial photography aesthetics."
        )
        
        logger.debug(f"ControlNet prompt built: {prompt[:100]}...")
        return prompt
    
    def build_nanobanan_prompt(self, background_description, style_keywords=None,
                              exclude_product=True):
        """
        Build prompt for Nano Banana background generation
        
        This should ONLY describe the background (no product)
        
        Args:
            background_description: Desired background scene
            style_keywords: List of style keywords
            exclude_product: Ensure product is not in prompt
            
        Returns:
            Background-only prompt
        """
        
        if style_keywords is None:
            style_keywords = ["vibrant", "Instagram-worthy", "Gen-Z aesthetic"]
        
        style_str = ", ".join(style_keywords)
        
        prompt = (
            f"A creative and visually stunning background scene featuring {background_description}. "
            f"Style: {style_str}. "
            f"High quality commercial photography background with perfect lighting and composition. "
        )
        
        if exclude_product:
            prompt += "Background only, no products, no text, no objects in foreground. "
        
        logger.debug(f"Nano Banana prompt built: {prompt[:100]}...")
        return prompt


class SmartPromptEnhancer:
    """
    Uses Gemini to enhance prompts
    Creates hyper-specific, optimized prompts
    """
    
    def __init__(self, api_key=None):
        import os
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("No Gemini API key for prompt enhancement")
            self.model = None
            self._initialized = False
            return
        
        # DON'T initialize yet - lazy loading
        self.model = None
        self._initialized = False
        logger.info("SmartPromptEnhancer created (will initialize on first use)")
    
    def _initialize(self):
        """Initialize Gemini (lazy)"""
        if self._initialized or not self.api_key:
            return
        
        try:
            import google.generativeai as genai
            
            logger.info("Initializing Gemini for prompt enhancement...")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self._initialized = True
            logger.info("✅ Gemini initialized for prompt enhancement")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini: {str(e)}")
            self.model = None
    
    def enhance_for_controlnet(self, product_name, background_idea, 
                               product_features=None, target_audience="Gen-Z"):
        """Generate hyper-specific descriptive prompt for ControlNet"""
        
        if not self._initialized:
            self._initialize()
        
        if not self.model:
            logger.warning("Gemini not available, returning basic prompt")
            return f"This photo showcases {product_name}. The product is clearly highlighted and placed in {background_idea}."
        
        features_text = ""
        if product_features:
            features_text = f"Product features: {', '.join(product_features)}. "
        
        enhancement_request = f"""
You are an expert in creating prompts for AI image generation using ControlNet inpainting.

Create a DESCRIPTIVE (not instructional) prompt that describes an ENTIRE product photograph including both the product and background.

Product: {product_name}
{features_text}
Background concept: {background_idea}
Target audience: {target_audience}

Requirements:
1. Use descriptive language (e.g., "This photo showcases...")
2. Describe the ENTIRE scene (product + background together)
3. Include lighting, mood, and atmosphere
4. Emphasize product clarity and appeal
5. Make it commercial/social media worthy
6. Keep it under 150 words

Generate the prompt now:
"""
        
        try:
            response = self.model.generate_content(
                enhancement_request,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                }
            )
            
            enhanced_prompt = response.text.strip()
            logger.info(f"Enhanced ControlNet prompt generated")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            return f"This photo showcases {product_name}. The product is clearly highlighted and placed in {background_idea}."
    
    def enhance_for_nanobanan(self, background_concept, style_preferences=None,
                             color_palette=None):
        """Generate hyper-specific prompt for Nano Banana background generation"""
        
        if not self._initialized:
            self._initialize()
        
        if not self.model:
            logger.warning("Gemini not available, returning basic prompt")
            return f"A creative background featuring {background_concept}. Background only, no products."
        
        style_text = ""
        if style_preferences:
            style_text = f"Style preferences: {', '.join(style_preferences)}. "
        
        color_text = ""
        if color_palette:
            color_text = f"Color palette: {', '.join(color_palette)}. "
        
        enhancement_request = f"""
Create a hyper-specific prompt for generating a BACKGROUND ONLY (no products) for product photography.

Background concept: {background_concept}
{style_text}
{color_text}

Requirements:
1. Describe ONLY the background (no products, no foreground objects)
2. Be extremely specific about colors, lighting, composition
3. Include atmospheric details and mood
4. Make it Gen-Z appealing and Instagram-worthy
5. Keep it concise (under 100 words)

Generate the background prompt now:
"""
        
        try:
            response = self.model.generate_content(
                enhancement_request,
                generation_config={
                    'temperature': 0.8,
                    'top_p': 0.9,
                    'top_k': 40,
                }
            )
            
            enhanced_prompt = response.text.strip()
            logger.info(f"Enhanced Nano Banana prompt generated")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            return f"Creative background for product photography: {background_concept}"