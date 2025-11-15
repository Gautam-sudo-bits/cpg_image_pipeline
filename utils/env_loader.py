"""
Environment variable loader
Loads API keys and configuration from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import logger

class EnvLoader:
    """Load and manage environment variables"""
    
    def __init__(self):
        # Load .env file from project root
        env_path = Path(__file__).parent.parent / '.env'
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            logger.warning(f".env file not found at: {env_path}")
            logger.warning("API key must be set via command line or environment variable")
    
    @staticmethod
    def get_api_key(env_var_name='GEMINI_API_KEY', required=False):
        """
        Get API key from environment
        
        Args:
            env_var_name: Name of environment variable
            required: If True, raise error if not found
            
        Returns:
            API key string or None
        """
        api_key = os.getenv(env_var_name)
        
        if api_key:
            # Mask key for logging
            masked_key = api_key[:8] + '...' + api_key[-4:] if len(api_key) > 12 else '***'
            logger.info(f"API key loaded: {masked_key}")
            return api_key
        else:
            if required:
                logger.error(f"API key '{env_var_name}' not found in environment!")
                logger.error("Please set it in .env file or via command line")
                raise ValueError(f"Required API key '{env_var_name}' not found")
            else:
                logger.warning(f"API key '{env_var_name}' not found - using fallback mode")
                return None
    
    @staticmethod
    def get_model_name(env_var_name, default):
        """Get model name from environment or use default"""
        return os.getenv(env_var_name, default)

# Global instance
env_loader = EnvLoader()