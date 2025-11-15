"""
Automatic prompt generation when user doesn't provide one
Uses Gemini to analyze image and create suitable prompts
PRIORITIZES LLM - fallback only when API unavailable
"""
import os
from PIL import Image
import google.generativeai as genai
from utils.logger import logger
from utils.env_loader import env_loader

class AutoPromptGenerator:
    """
    Generates creative prompts automatically by analyzing the product image
    """
    
    def __init__(self, api_key=None, config=None):
        # Priority: passed api_key > .env > config
        if api_key:
            self.api_key = api_key
        else:
            # Load from .env
            env_var = config.get('api', {}).get('api_key_env_var', 'GEMINI_API_KEY') if config else 'GEMINI_API_KEY'
            require_key = config.get('api', {}).get('require_api_key', False) if config else False
            self.api_key = env_loader.get_api_key(env_var, required=require_key)
        
        self.config = config
        self.model = None
        self._model_initialized = False
        
        # Check if LLM should be used
        use_llm = config.get('api', {}).get('use_llm_prompt_generation', True) if config else True
        
        if not self.api_key:
            if use_llm:
                logger.error("LLM prompt generation is ENABLED but no API key found!")
                logger.error("Please add GEMINI_API_KEY to .env file")
                logger.warning("Falling back to template prompts")
            else:
                logger.info("LLM prompt generation disabled in config")
        else:
            logger.info("LLM prompt generation ENABLED with API key")
    
    def _initialize_model(self):
        """Initialize Gemini model (lazy loading)"""
        if self._model_initialized or not self.api_key:
            return
        
        try:
            logger.info("Initializing Gemini for LLM prompt generation...")
            genai.configure(api_key=self.api_key)
            
            model_name = self.config.get('prompt_enhancement', {}).get('model', 'gemini-2.5-pro') if self.config else 'gemini-2.5-flash'
            self.model = genai.GenerativeModel(model_name)
            
            self._model_initialized = True
            logger.info(f"[SUCCESS] Gemini model initialized: {model_name}")
        except Exception as e:
            logger.error(f"[FAILED] Could not initialize Gemini: {str(e)}")
            self.model = None
    
    def analyze_and_generate_prompts(self, image):
        """
        Analyze product image and generate prompts
        PRIORITIZES LLM - only falls back if API fails
        
        Args:
            image: PIL Image object
            
        Returns:
            dict with 'product_description' and 'background_prompt'
        """
        
        # Try to initialize model if we have API key
        if not self._model_initialized and self.api_key:
            self._initialize_model()
        
        # If model available, USE IT (primary mode)
        if self.model is not None:
            try:
                return self._generate_with_llm(image)
            except Exception as e:
                logger.error(f"LLM prompt generation failed: {str(e)}")
                logger.warning("Falling back to template prompts")
                return self._generate_creative_fallback()
        else:
            # No API key - must use fallback
            logger.warning("No LLM available - using template prompts")
            return self._generate_creative_fallback()
    
    def _generate_with_llm(self, image):
        """Generate prompts using Gemini LLM (PRIMARY METHOD)"""
        
        logger.info("Generating prompts with Gemini LLM...")
        
        analysis_prompt = """
Analyze this product image and provide a concise description.

CRITICAL RULES:
1. Keep TOTAL response under 100 words
2. NO markdown formatting (no *, **, bullets, etc.)
3. Do NOT read or mention specific text/brand names on the product
4. Describe product GENERICALLY (e.g., "yellow beverage can" not specific brand)
5. Be concise and direct
6. Focus on: product type, colors, design style

PRODUCT DESCRIPTION (max 30 words):
Describe the product type, shape, primary colors, overall design. Generic terms only.

BACKGROUND IDEA (max 50 words):
Suggest a creative, Gen-Z appealing background that complements the product. Be specific about scene, colors, mood but concise.

STYLE (max 20 words):
Overall aesthetic style in adjectives.

Format EXACTLY as:
PRODUCT: [description]
BACKGROUND: [suggestion]
STYLE: [style words]
"""
        
        response = self.model.generate_content(
            [analysis_prompt, image],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        
        response_text = response.text
        logger.debug(f"LLM response: {response_text[:150]}...")
        
        # Parse response
        product_desc = self._extract_section(response_text, "PRODUCT:")
        background_idea = self._extract_section(response_text, "BACKGROUND:")
        style_hint = self._extract_section(response_text, "STYLE:")
        
        # Clean text
        product_desc = self._clean_text(product_desc) if product_desc else "modern product"
        background_idea = self._clean_text(background_idea) if background_idea else "creative vibrant scene"
        style_hint = self._clean_text(style_hint) if style_hint else "modern, clean"
        
        result = {
            'product_description': product_desc,
            'background_prompt': background_idea,
            'style_suggestion': style_hint,
            'full_analysis': response_text,
            'source': 'llm'
        }
        
        logger.info(f"[LLM] Product: {product_desc[:40]}...")
        logger.info(f"[LLM] Background: {background_idea[:40]}...")
        
        return result
    
    def _extract_section(self, text, section_marker):
        """Extract content after a section marker"""
        try:
            if section_marker in text:
                start = text.index(section_marker) + len(section_marker)
                next_sections = ['\nPRODUCT:', '\nBACKGROUND:', '\nSTYLE:']
                end = len(text)
                for marker in next_sections:
                    if marker in text[start:]:
                        potential_end = text.index(marker, start)
                        end = min(end, potential_end)
                
                content = text[start:end].strip()
                return content
            return None
        except Exception as e:
            logger.debug(f"Error extracting {section_marker}: {str(e)}")
            return None
    
    def _clean_text(self, text):
        """Remove markdown and formatting artifacts"""
        if not text:
            return text
        
        text = text.replace('**', '').replace('*', '')
        text = text.replace('###', '').replace('##', '').replace('#', '')
        text = text.replace('- ', '').replace('• ', '')
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _generate_creative_fallback(self):
        """Template-based fallback (ONLY when LLM unavailable)"""
        import random
        
        logger.warning("[FALLBACK] Using template prompts (LLM not available)")
        
        product_types = [
            "modern product with vibrant design",
            "colorful packaged consumer product",
            "stylish beverage container with bold branding",
            "eye-catching retail product",
            "contemporary packaged item with bright colors"
        ]
        
        background_ideas = [
            "vibrant gradient with neon geometric shapes",
            "soft pastel bokeh with dreamy lighting",
            "urban street art with colorful graffiti",
            "tropical paradise with bright sunny vibes",
            "minimalist studio with dramatic lighting",
            "futuristic neon-lit scene",
            "organic botanical setting with soft light",
            "pop art style with bold dynamic colors"
        ]
        
        styles = [
            "modern, Instagram-worthy, trendy",
            "clean, professional, commercial",
            "vibrant, energetic, Gen-Z",
            "sophisticated, elegant, upscale",
            "bold, creative, eye-catching"
        ]
        
        result = {
            'product_description': random.choice(product_types),
            'background_prompt': random.choice(background_ideas),
            'style_suggestion': random.choice(styles),
            'full_analysis': 'Template fallback',
            'source': 'fallback'
        }
        
        return result
    
    def generate_for_method1(self, image):
        """Generate descriptive prompt for Method 1"""
        analysis = self.analyze_and_generate_prompts(image)
        
        prompt = (
            f"This photo showcases {analysis['product_description']}. "
            f"The product is clearly highlighted and prominently displayed, "
            f"placed in {analysis['background_prompt']}. "
            f"The overall image maintains a {analysis['style_suggestion']} visual style, "
            f"with professional commercial photography lighting."
        )
        
        if len(prompt) > 600:
            prompt = prompt[:600]
        
        return prompt, analysis
    
    def generate_for_method2(self, image):
        """Generate background-only prompt for Method 2"""
        analysis = self.analyze_and_generate_prompts(image)
        
        prompt = (
            f"A creative background scene featuring {analysis['background_prompt']}. "
            f"Style: {analysis['style_suggestion']}. "
            f"High quality commercial photography background with perfect lighting. "
            f"Background only, no products, no text, no foreground objects."
        )
        
        return prompt, analysis