"""
Nano Banana API handler using Gemini 2.5 Flash Image Generation
Following the WORKING pattern from user's code
"""
import os
import io
from PIL import Image
import google.generativeai as genai
from utils.logger import logger
from utils.env_loader import env_loader

class NanoBananaAPI:
    def __init__(self, config, api_key=None):
        self.config = config
        self.nb_config = config.get('nano_banana', {})
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            env_var = config.get('api', {}).get('api_key_env_var', 'GEMINI_API_KEY')
            require_key = self.nb_config.get('require_api_key', True)
            self.api_key = env_loader.get_api_key(env_var, required=require_key)
        
        if not self.api_key:
            logger.error("Nano Banana requires Gemini API key!")
            raise ValueError("Add GEMINI_API_KEY to .env file")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Model name - use gemini-2.5-flash-image for image generation
        self.model_name = self.nb_config.get('model', 'gemini-2.5-flash-image')
        self.model = None
        self.text_model = None
        self._initialized = False
        
        logger.info(f"NanoBananaAPI initialized")
    
    def _initialize(self):
        """Initialize Gemini models"""
        if self._initialized:
            return
        
        try:
            logger.info(f"Initializing Gemini image generation model: {self.model_name}")
            
            # Image generation model
            self.model = genai.GenerativeModel(self.model_name)
            
            # Text model for prompt enhancement (can use different model)
            text_model_name = self.config.get('prompt_enhancement', {}).get('model', 'gemini-2.0-flash-thinking-exp-1219')
            self.text_model = genai.GenerativeModel(text_model_name)
            
            self._initialized = True
            logger.info(f"[SUCCESS] Gemini models initialized")
            logger.info(f"  Image: {self.model_name}")
            logger.info(f"  Text: {text_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise
    
    def generate_background(self, prompt, reference_image=None, 
                       width=1024, height=1024, num_images=1):
        """Generate background using Gemini 2.5 Flash Image"""
        
        if not self._initialized:
            self._initialize()
        
        try:
            logger.info(f"Generating background with Gemini {self.model_name}...")
            
            test_prompt = """
        Create a vibrant outdoor picnic scene background. The main element is a pink and purple checkered picnic blanket spread on bright green grass. Scattered on and around the blanket are colorful faux flowers in hot pink, yellow, and orange. A pair of trendy sunglasses with neon frames sits on the blanket. The scene is photographed from slightly above at a 45-degree angle. Bright natural sunlight creates soft shadows. The atmosphere is playful, youthful, and Instagram-worthy. The background should be sharp and detailed with natural bokeh in the distance showing more grass and flowers. Commercial photography quality, vibrant colors, Gen-Z aesthetic. Background only - no products, no text.
        """
            
            logger.info("USING TEST PROMPT:")
            logger.info(test_prompt)
            
            contents = [test_prompt]  # Use test instead of passed prompt
            
            response = self.model.generate_content(contents)
            # Log usage
            if hasattr(response, 'usage_metadata'):
                cached_tokens = response.usage_metadata.cached_content_token_count
                logger.debug(f"Cached tokens: {cached_tokens}")
            
            # Extract image from response (following user's working pattern)
            generated_image_bytes = None
            
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Extract bytes from inline_data
                        generated_image_bytes = part.inline_data.data
                        logger.info("✓ Successfully extracted image bytes from inline_data")
                        break
            
            if generated_image_bytes:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(generated_image_bytes))
                
                # Resize if needed
                if image.size != (width, height):
                    logger.info(f"Resizing from {image.size} to {width}x{height}")
                    image = image.resize((width, height), Image.LANCZOS)
                
                logger.info(f"[SUCCESS] Background generated: {width}x{height}")
                return image
            else:
                # No image found in response
                logger.error("No inline_data found in Gemini response")
                logger.debug(f"Response structure: {response}")
                raise ValueError("No image data in Gemini response")
        
        except Exception as e:
            logger.error(f"Background generation failed: {str(e)}")
            logger.error(f"Full error details:", exc_info=True)
            raise
    
    def enhance_prompt(self, base_prompt, context="background generation", style_keywords=None):
        """Use Gemini to enhance the prompt"""
        
        if not self._initialized:
            self._initialize()
        
        try:
            style_text = ""
            if style_keywords:
                style_text = f"Style keywords: {', '.join(style_keywords)}. "
            
            enhancement_request = (
                f"Enhance this image generation prompt to be hyper-specific and visually detailed.\n\n"
                f"Context: {context}\n"
                f"Base prompt: {base_prompt}\n"
                f"{style_text}\n\n"
                f"Requirements:\n"
                f"- Make it extremely specific about colors, lighting, composition\n"
                f"- Include atmospheric details and mood\n"
                f"- Gen-Z appealing and Instagram-worthy\n"
                f"- Concise (under 100 words)\n"
                f"- For backgrounds: NO products, NO text, NO foreground objects\n\n"
                f"Enhanced prompt:"
            )
            
            response = self.text_model.generate_content(
                enhancement_request,
                generation_config=genai.GenerationConfig(temperature=0.7)
            )
            
            enhanced = response.text.strip().strip('"\'')
            logger.info(f"[SUCCESS] Prompt enhanced with Gemini")
            logger.debug(f"Enhanced: {enhanced[:100]}...")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Prompt enhancement failed: {str(e)}")
            return base_prompt