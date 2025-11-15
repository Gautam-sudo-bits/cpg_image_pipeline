# CPG Image Generation Pipeline 📦✨

Generate commercial-grade Consumer Packaged Goods (CPG) product images with AI-generated backgrounds while maintaining **100% product text and detail preservation**.

## 🎯 Problem Statement

Current AI image generation models struggle with:
- ❌ Text/typography distortion on product labels
- ❌ Loss of product details and features
- ❌ Inconsistent brand representation

## 💡 Solution

This pipeline uses a **smart workaround**:
1. Start with **real product photos** (iPhone quality)
2. Extract product as foreground using **rembg**
3. Generate **only the background** using AI
4. Product pixels remain **untouched and perfect**

---

## 🚀 Features

### Two Generation Methods

#### **Method 1: Pure ControlNet Inpainting**
- Uses Qwen-Image-ControlNet for background inpainting
- AI respects product mask completely
- Single-pass generation
- Best for: Quick, high-quality results

#### **Method 2: Nano Banana + Composite**
- Generates background separately with Gemini (Nano Banana)
- Composites real product on top
- Zero risk of product corruption
- Best for: Maximum product preservation, creative backgrounds

### Key Capabilities
- ✅ **100% Text Preservation** - Product labels stay pixel-perfect
- ✅ **HEIC to PNG Conversion** - Maintains iPhone photo quality
- ✅ **GPU Acceleration** - Automatic CPU fallback
- ✅ **Style Presets** - Gen-Z optimized aesthetics
- ✅ **Stage Visualization** - See every processing step
- ✅ **Modular Architecture** - Easy to extend

---

## 📁 Project Structure

cpg-image-creation/
├── config/
│ ├── config.yaml # Main configuration
│ └── style_presets.yaml # Style presets (Gen-Z, minimal, etc.)
├── modules/
│ ├── image_loader.py # HEIC/PNG loading & conversion
│ ├── foreground_extractor.py # rembg-based extraction
│ ├── mask_processor.py # Mask refinement
│ ├── compositing_engine.py # Advanced compositing
│ └── visualizer.py # Pipeline visualization
├── methods/
│ ├── method1_controlnet_inpaint.py # Method 1 implementation
│ └── method2_nanobanan_composite.py # Method 2 implementation
├── models/
│ ├── nano_banana_api.py # Gemini API handler
│ └── controlnet_handler.py # ControlNet pipeline
├── utils/
│ ├── gpu_checker.py # GPU/CPU detection
│ └── logger.py # Logging system
├── input/ # Place your product images here
├── output/ # Generated images
├── main.py # Main entry point
├── requirements.txt
└── README.md

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd cpg-image-creation
2. Create Virtual Environment
Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Install Qwen ControlNet (if not auto-installed)
Bash

pip install git+https://github.com/huggingface/diffusers
5. Set Up API Key (for Method 2)
Bash

export GEMINI_API_KEY="your-gemini-api-key-here"
Or create a .env file:

text

GEMINI_API_KEY=your-key-here
🎮 Usage
Basic Usage
Method 1: ControlNet Inpainting
Bash

python main.py input/my_product.heic \
    --method method1_controlnet \
    --prompt "vibrant gradient background, modern aesthetic" \
    --style vibrant_genz
Method 2: Nano Banana + Composite
Bash

python main.py input/my_product.heic \
    --method method2_nanobanan \
    --prompt "summer beach vibes, tropical atmosphere" \
    --style summer_beach \
    --api-key YOUR_GEMINI_KEY


Advanced Usage:

Generate Multiple Variations
Bash

python main.py input/soda_can.heic \
    --method method2_nanobanan \
    --variations 5 \
    --prompt "energetic, Gen-Z appealing background"

Custom Output Path
Bash

python main.py input/product.jpg \
    --method method1_controlnet \
    --output output/custom_name.png

Use Custom Config
Bash

python main.py input/product.png \
    --config config/my_custom_config.yaml

Required:
  --input, -i       Path to input image (HEIC, PNG, JPG)
  --prompt, -p      Background generation prompt

Optional:
  --method, -m      Method: method1_controlnet, method2_nanobanan, both
  --style, -s       Style preset name
  --output, -o      Custom output path
  --config, -c      Custom config file path
  --api-key         Gemini API key
  --variations, -v  Number of variations (Method 2 only)