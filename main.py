import os
import sys
import argparse
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env FIRST (before any other imports)
from utils.env_loader import env_loader

from utils.logger import logger
from methods import Method1ControlNetInpaint, Method2NanoBananaComposite
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import logger first to see what's happening
try:
    from utils.logger import logger
    logger.info("Logger imported successfully")
except Exception as e:
    print(f"ERROR importing logger: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try importing methods with detailed error reporting
try:
    logger.info("Importing methods...")
    from methods import Method1ControlNetInpaint, Method2NanoBananaComposite
    logger.info("Methods imported successfully")
except Exception as e:
    logger.error(f"ERROR importing methods: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def setup_directories(config):
    """Create necessary directories"""
    paths = config.get('paths', {})
    
    dirs_to_create = [
        paths.get('input_dir', 'input'),
        paths.get('output_dir', 'output'),
        paths.get('temp_dir', 'temp'),
        paths.get('visualization_dir', 'output/visualizations'),
        'logs'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directories setup complete")

def main():
    """Main execution function"""
    
    try:
        logger.info("Starting main function...")
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='CPG Product Image Generator with Perfect Text Preservation',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Minimal - Just provide image (auto-generates everything!)
  python main.py -i input/soda.heic
  
  # With custom prompt
  python main.py -i input/bottle.png -p "tropical beach vibes"
  
  # Specify method and style
  python main.py -i input/product.jpg -m method1_controlnet -s vibrant_genz
  
  # Full control
  python main.py -i input/can.heic \\
    -p "urban rooftop at sunset" \\
    --product "energy drink can with bold graphics" \\
    --color "orange, purple, electric blue" \\
    -m method2_nanobanan -v 3
            """
        )
        parser.add_argument(
            '--input', '-i',
            type=str,
            required=True,
            help='Path to input product image (any format: HEIC, PNG, JPG, etc.)'
        )
        parser.add_argument(
            '--prompt', '-p',
            type=str,
            default=None,
            help='Prompt for background/scene (OPTIONAL - auto-generates if not provided)'
        )
        parser.add_argument(
            '--product', 
            type=str,
            default=None,
            help='Product description (OPTIONAL - auto-detects if not provided)'
        )
        parser.add_argument(
            '--method', '-m',
            type=str,
            choices=['method1_controlnet', 'method2_nanobanan', 'both'],
            default=None,
            help='Generation method (overrides config)'
        )
        parser.add_argument(
            '--style', '-s',
            type=str,
            default=None,
            help='Style preset name from style_presets.yaml'
        )
        parser.add_argument(
            '--color',
            type=str,
            default=None,
            help='Color palette (comma-separated, e.g., "blue, purple, pink")'
        )
        parser.add_argument(
            '--output', '-o',
            type=str,
            default=None,
            help='Output path for generated image'
        )
        parser.add_argument(
            '--config', '-c',
            type=str,
            default='config/config.yaml',
            help='Path to configuration file'
        )
        parser.add_argument(
            '--api-key',
            type=str,
            default=None,
            help='Gemini API key (or set GEMINI_API_KEY env var)'
        )
        parser.add_argument(
            '--variations', '-v',
            type=int,
            default=1,
            help='Number of variations to generate'
        )
        parser.add_argument(
            '--no-enhance',
            action='store_true',
            help='Disable automatic prompt enhancement'
        )
        
        args = parser.parse_args()
        logger.info(f"Arguments parsed: {args}")
        
        # Print banner
        print("\n" + "="*70)
        print("  CPG PRODUCT IMAGE GENERATOR")
        print("  Perfect Text Preservation + Creative Backgrounds")
        print("  Auto-generates prompts if not provided!")
        print("="*70 + "\n")
        
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Override prompt enhancement if disabled
        if args.no_enhance:
            config['prompt_enhancement']['enabled'] = False
            config['prompt_enhancement']['use_smart_enhancer'] = False
            logger.info("Prompt enhancement disabled")
        
        # Setup directories
        logger.info("Setting up directories...")
        setup_directories(config)
        
        # Determine method
        method = args.method if args.method else config.get('method', 'method1_controlnet')
        
        # Parse color palette
        color_palette = None
        if args.color:
            color_palette = [c.strip() for c in args.color.split(',')]
        
        # Log what we're doing
        logger.info(f"Input image: {args.input}")
        if args.prompt:
            logger.info(f"Prompt: {args.prompt}")
        else:
            logger.info("Prompt: AUTO-GENERATE")
        if args.product:
            logger.info(f"Product: {args.product}")
        else:
            logger.info("Product: AUTO-DETECT")
        logger.info(f"Method: {method}")
        if args.style:
            logger.info(f"Style preset: {args.style}")
        if color_palette:
            logger.info(f"Color palette: {color_palette}")
        
        # Check if input file exists
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            print(f"\nError: File not found: {args.input}\n")
            return
        
        # Execute based on method
        if method == 'method1_controlnet':
            logger.info(" Starting Method 1: Pure ControlNet Inpainting\n")
            
            method1 = Method1ControlNetInpaint(config, api_key=args.api_key)
            
            if args.variations > 1:
                result_paths = method1.generate_with_variations(
                    input_image_path=args.input,
                    prompt=args.prompt,
                    product_description=args.product,
                    num_variations=args.variations
                )
                print(f"\nSuccess! Generated {len(result_paths)} variations:")
                for path in result_paths:
                    print(f"  - {path}")
                print()
            else:
                result_path = method1.generate(
                    input_image_path=args.input,
                    prompt=args.prompt,
                    product_description=args.product,
                    style_preset=args.style,
                    output_path=args.output
                )
                print(f"\nSuccess! Generated image saved to: {result_path}\n")
        
        elif method == 'method2_nanobanan':
            logger.info(" Starting Method 2: Nano Banana + Composite\n")
            
            method2 = Method2NanoBananaComposite(config, api_key=args.api_key)
            
            if args.variations > 1:
                # Generate multiple variations
                prompts_list = [args.prompt] * args.variations if args.prompt else None
                result_paths = method2.generate_multiple_variations(
                    input_image_path=args.input,
                    prompts_list=prompts_list,
                    style_preset=args.style,
                    color_palette=color_palette
                )
                print(f"\nSuccess! Generated {len(result_paths)} variations:")
                for path in result_paths:
                    print(f"  - {path}")
                print()
            else:
                result_path = method2.generate(
                    input_image_path=args.input,
                    prompt=args.prompt,
                    style_preset=args.style,
                    color_palette=color_palette,
                    output_path=args.output
                )
                print(f"\nSuccess! Generated image saved to: {result_path}\n")
        
        elif method == 'both':
            logger.info("\nRunning BOTH methods for comparison\n")
            
            # Method 1
            logger.info("\n--- Method 1: ControlNet ---")
            method1 = Method1ControlNetInpaint(config, api_key=args.api_key)
            result1 = method1.generate(
                input_image_path=args.input,
                prompt=args.prompt,
                product_description=args.product,
                style_preset=args.style
            )
            
            # Method 2
            logger.info("\n--- Method 2: Nano Banana ---")
            method2 = Method2NanoBananaComposite(config, api_key=args.api_key)
            result2 = method2.generate(
                input_image_path=args.input,
                prompt=args.prompt,
                style_preset=args.style,
                color_palette=color_palette
            )
            
            print("\nSuccess! Both methods completed:")
            print(f"  Method 1: {result1}")
            print(f"  Method 2: {result2}\n")
        
        else:
            logger.error(f"Unknown method: {method}")
            return
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        print(f"\nError: {str(e)}\n")
        traceback.print_exc()
        return

if __name__ == "__main__":
    try:
        logger.info("=== CPG Image Generator Starting ===")
        main()
        logger.info("=== CPG Image Generator Finished ===")
    except Exception as e:
        print(f" Fatal error: {e}\n")
        traceback.print_exc()
        sys.exit(1)