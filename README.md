# SuperCharged Photo Editor

A modern, AI-powered photo editing toolkit for web and mobile apps. Process AI-generated images with background removal, **AI upscaling**, vectorization, and perfect resizing - all in one place.

## Features

### AI-Powered Image Upscaling (Real-ESRGAN)
**The magic for small AI-generated images!** Upscale your Gemini, DALL-E, Midjourney, or any AI art 2x-4x **without losing quality**. The AI actually *adds* detail instead of blur!

```bash
# Upscale a small Gemini image 4x (512px -> 2048px)
python photo_editor.py gemini_art.png --upscale

# Use anime model for illustrations
python photo_editor.py illustration.png --upscale --upscale-model anime
```

| Model | Best For | Scale |
|-------|----------|-------|
| `general` | Photos, general images | 4x |
| `anime` | Anime, illustrations, AI art | 4x |
| `fast` | Quick processing | 4x |
| `--upscale-2x` | Less aggressive upscaling | 2x |

### AI-Powered Background Removal
Remove backgrounds instantly using deep learning (U2-Net model via rembg). Perfect for product photos, AI-generated art, and graphics.

### Raster to Vector Conversion
Convert any image to scalable SVG using vtracer. Multiple presets for photos, illustrations, logos, and pixel art.

### Smart Resizing
High-quality resizing with multiple algorithms (LANCZOS, BICUBIC, BILINEAR).

### Drop Folder Watcher
Drop images into a folder and watch them get processed automatically. Perfect for batch workflows.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# AI UPSCALE - Make small AI images HUGE without blur!
python photo_editor.py small_image.png --upscale

# Process a single image (full pipeline with upscaling)
python photo_editor.py image.png --pipeline --upscale

# Remove background only
python photo_editor.py image.png --remove-bg

# Convert to vector SVG
python photo_editor.py image.png --vectorize

# Resize to specific dimensions
python photo_editor.py image.png --resize 1024 1024

# Start the drop folder watcher
python drop_watcher.py

# FORMAT CONVERTER - Make banners, buttons, postcards, etc.
python format_converter.py image.png banner
python format_converter.py image.png postcard flyer icon
python format_converter.py --list  # See all presets
```

## Format Converter - "Make This a Banner/Button/Postcard"

Convert your AI-generated images to any format with one command:

```bash
# Single format
python format_converter.py my_art.png banner

# Multiple formats at once
python format_converter.py logo.png icon favicon app_icon button

# All social media sizes
python format_converter.py promo.png --all-social

# All print sizes (postcards, flyers, posters)
python format_converter.py artwork.png --all-print

# Custom size
python format_converter.py image.png --custom 800x600

# List all available presets
python format_converter.py --list
```

### Available Presets

| Shortcut | Format | Size | Use Case |
|----------|--------|------|----------|
| `banner` | Twitter Banner | 1500x500 | Social media header |
| `button` | Button Medium | 200x60 | Web buttons |
| `icon` | Icon | 256x256 | App/web icons |
| `thumbnail` | Thumbnail | 300x300 | Preview images |
| `postcard` | Postcard 6x4 | 1800x1200 | Print postcards |
| `flyer` | Flyer Letter | 2550x3300 | Print flyers |
| `poster` | Poster 11x17 | 3300x5100 | Print posters |
| `hero` | Hero Image | 1920x1080 | Website heroes |

**Social Media**: `twitter_banner`, `twitter_post`, `instagram`, `instagram_story`, `facebook_cover`, `linkedin_banner`, `youtube_thumbnail`

**Print (300 DPI)**: `postcard`, `postcard_5x7`, `flyer`, `flyer_a4`, `business_card`, `poster`, `poster_18x24`

**Web/App**: `button_small`, `button_large`, `icon_64`, `icon_128`, `icon_512`, `app_icon`, `favicon`, `hero`, `og_image`

### Output Organization

```
output/
├── social/
│   ├── twitter/
│   │   └── image_twitter_banner.png
│   ├── instagram/
│   └── facebook/
├── print/
│   ├── postcards/
│   ├── flyers/
│   └── posters/
└── web/
    ├── buttons/
    ├── icons/
    └── thumbnails/
```

## Usage Examples

### Python API

```python
from photo_editor import PhotoEditor, VectorMode, ResizeMode, UpscaleModel

editor = PhotoEditor()

# AI UPSCALE - The magic for small AI images!
result = editor.ai_upscale("gemini_512.png", "gemini_2048.png", scale=4)
print(f"Upscaled to: {result.output_size}")  # (2048, 2048)

# Use anime model for illustrations
result = editor.ai_upscale(
    "illustration.png",
    model=UpscaleModel.ANIME_X4
)

# Remove background from AI-generated image
result = editor.remove_background("ai_art.png", "ai_art_nobg.png")
print(f"Saved to: {result.output_path}")

# Convert to vector SVG
result = editor.vectorize("logo.png", "logo.svg", mode=VectorMode.LOGO)

# Smart resize with LANCZOS
result = editor.smart_resize(
    "photo.png",
    "photo_resized.png",
    width=1024,
    height=1024,
    mode=ResizeMode.LANCZOS
)

# Full pipeline: remove bg -> vectorize -> resize
results = editor.process_full_pipeline(
    "input.png",
    "output/",
    remove_bg=True,
    create_vector=True,
    resize_config={'sizes': [
        {'width': 512, 'height': 512, 'suffix': '_small'},
        {'width': 1024, 'height': 1024, 'suffix': '_medium'},
        {'width': 2048, 'height': 2048, 'suffix': '_large'},
    ]}
)

# Batch process a directory
results = editor.batch_process("./my_images/", "./output/")
```

### Quick Functions

```python
from photo_editor import quick_remove_bg, quick_vectorize, quick_resize, quick_upscale

# One-liners for common operations
quick_upscale("gemini_art.png")        # -> gemini_art_upscaled_4x.png (THE MAGIC!)
quick_upscale("anime.png", model="anime")  # Use anime model
quick_remove_bg("photo.png")           # -> photo_nobg.png

### REST API - For Web & Mobile Apps

Start the API server and call from any language/platform:

```bash
# Start the server
python api_server.py

# Server runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upscale` | POST | AI upscale image (Real-ESRGAN) |
| `/remove-bg` | POST | Remove background |
| `/vectorize` | POST | Convert to SVG |
| `/resize` | POST | Smart resize |
| `/convert/{preset}` | POST | Convert to preset (banner, postcard, etc.) |
| `/pipeline` | POST | Full processing pipeline |
| `/presets` | GET | List all available presets |
| `/batch/convert` | POST | Batch convert multiple images |

**Example - cURL:**
```bash
# Upscale an image 4x
curl -X POST "http://localhost:8000/upscale?scale=4&model=anime" \
  -F "file=@my_art.png" -o upscaled.png

# Remove background
curl -X POST "http://localhost:8000/remove-bg" \
  -F "file=@photo.jpg" -o nobg.png

# Convert to Twitter banner
curl -X POST "http://localhost:8000/convert/banner" \
  -F "file=@image.png" -o banner.png
```

**Example - JavaScript/Fetch:**
```javascript
// Upscale an image
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/upscale?scale=4', {
  method: 'POST',
  body: formData
});

const blob = await response.blob();
const url = URL.createObjectURL(blob);
```

**Example - Python Requests:**
```python
import requests

# Remove background
with open("photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/remove-bg",
        files={"file": f}
    )

with open("nobg.png", "wb") as f:
    f.write(response.content)
quick_vectorize("logo.png")            # -> logo.svg
quick_resize("image.png", width=512)   # -> image_resized.png
```

### Drop Folder Watcher

```bash
# Basic usage - watches ./drop folder
python drop_watcher.py

# Custom folders
python drop_watcher.py --watch ~/Downloads --output ~/processed

# Customize processing
python drop_watcher.py --sizes 256 512 1024 2048  # Multiple output sizes
python drop_watcher.py --no-vector                 # Skip SVG conversion
python drop_watcher.py --vector-mode logo          # Logo-optimized vectors

# Process existing files first, then watch
python drop_watcher.py --process-existing
```

## Command Line Reference

### photo_editor.py

```bash
# Single operations
python photo_editor.py INPUT --remove-bg [-o OUTPUT]
python photo_editor.py INPUT --vectorize [-o OUTPUT]
python photo_editor.py INPUT --resize WIDTH HEIGHT [-o OUTPUT]
python photo_editor.py INPUT --scale 2.0 [-o OUTPUT]

# Full pipeline
python photo_editor.py INPUT --pipeline [-o OUTPUT_DIR]

# Batch processing
python photo_editor.py INPUT_DIR --batch [-o OUTPUT_DIR]
```

### drop_watcher.py

```bash
python drop_watcher.py [OPTIONS]

Options:
  -w, --watch DIR         Folder to watch (default: ./drop)
  -o, --output DIR        Output folder (default: ./processed)
  -s, --sizes N [N ...]   Output sizes in pixels (default: 512 1024)
  -m, --vector-mode MODE  Vectorization style: photo, illustration, logo, pixel_art
  --no-bg-removal         Disable background removal
  --no-vector             Disable SVG vectorization
  -p, --process-existing  Process existing images before watching
```

## Libraries Used

| Library | Purpose |
|---------|---------|
| **Real-ESRGAN** | AI image upscaling - make small images HUGE without blur! |
| **rembg** | AI background removal using U2-Net deep learning |
| **vtracer** | Fast raster-to-vector conversion (Rust-based) |
| **Pillow** | Core image processing |
| **OpenCV** | Advanced resizing algorithms |
| **PyTorch** | Deep learning framework (powers Real-ESRGAN) |
| **watchdog** | Folder watching for auto-processing |
| **NumPy** | Numerical operations |

## Directory Structure

```
mini-photo-converter/
├── photo_editor.py          # Main photo editor module
├── drop_watcher.py          # Auto-processing folder watcher
├── requirements.txt         # Dependencies
├── README.md                # This file
│
├── # Legacy scripts (still functional)
├── optimize_images.py       # Basic image optimization
├── create_banner.py         # Twitter banner creation
├── create_letterbox_banner.py
├── reduced_background_images.py
│
└── # Output directories (created automatically)
    ├── drop/                # Drop folder for watcher
    └── processed/           # Processed output
        └── image_name/
            ├── image_name_nobg.png
            ├── image_name.svg
            ├── image_name_512.png
            └── image_name_1024.png
```

## Installation

### Apple Silicon (M1/M2/M3) - Recommended Setup

The photo editor fully supports Apple Silicon with **MPS GPU acceleration**. Your M-series chip will be used for fast AI processing!

```bash
# Clone the repository
git clone https://github.com/yourusername/mini-photo-converter.git
cd mini-photo-converter

# Create virtual environment with Python 3.10+ (required for MPS)
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with MPS support (Apple Silicon GPU)
pip install --upgrade pip
pip install torch torchvision

# Verify MPS is available (should print "True")
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Install all other dependencies
pip install -r requirements.txt

# Test the installation
python photo_editor.py --help
```

#### Expected Output on Apple Silicon
When you run upscaling, you should see:
```
[Real-ESRGAN] Using device: mps
```

#### Performance on Apple Silicon

| Chip | 512→2048 (4x) | 1024→4096 (4x) | RAM Needed |
|------|---------------|----------------|------------|
| M1   | ~5-8 sec      | ~20-30 sec     | 8GB+       |
| M2   | ~4-6 sec      | ~15-25 sec     | 8GB+       |
| M3   | ~3-5 sec      | ~12-20 sec     | 8GB+       |
| M3 Pro (36GB) | ~2-4 sec | ~10-15 sec  | Plenty!    |

### Windows / Linux with NVIDIA GPU

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch with CUDA (choose your CUDA version)
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### CPU Only (Any Platform)

```bash
python -m venv .venv
source .venv/bin/activate

# Standard PyTorch (CPU)
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt
```

Note: CPU is slower but works on any machine.

### Troubleshooting

**"MPS not available" on Mac:**
```bash
# Make sure you have Python 3.10+
python3 --version

# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision
```

**Memory errors on large images:**
```python
# Use tiling for large images
editor.ai_upscale("large.png", tile_size=512)
```

**Slow performance:**
- Close other GPU-intensive apps
- Check Activity Monitor → GPU usage
- Use `--upscale-model fast` for quicker results

### First Run Note
The first time you use AI features, models will be downloaded automatically:
- **Real-ESRGAN**: ~64MB per model (downloads on first upscale)
- **rembg/U2-Net**: ~170MB (downloads on first background removal)

Models are cached in `~/.cache/` and only download once.

## API Reference

### PhotoEditor Class

```python
class PhotoEditor:
    def __init__(self, default_output_dir="processed", rembg_model="u2net"):
        """
        rembg_model options:
        - "u2net" (default, best quality)
        - "u2netp" (faster, slightly lower quality)
        - "u2net_human_seg" (optimized for humans)
        - "isnet-general-use" (good general purpose)
        - "isnet-anime" (optimized for anime)
        """

    # AI Upscaling - THE MAGIC FOR SMALL IMAGES!
    def ai_upscale(self, input_path, output_path=None, scale=4,
                   model=UpscaleModel.GENERAL_X4, denoise_strength=0.5)

    def remove_background(self, input_path, output_path=None, alpha_matting=False)
    def vectorize(self, input_path, output_path=None, mode=VectorMode.ILLUSTRATION)
    def smart_resize(self, input_path, output_path=None, width=None, height=None,
                     scale=None, mode=ResizeMode.LANCZOS)
    def process_full_pipeline(self, input_path, output_dir=None, ai_upscale=False, ...)
    def batch_process(self, input_dir, output_dir=None, ...)
```

### Enums

```python
class UpscaleModel(Enum):
    GENERAL_X4 = "RealESRGAN_x4plus"      # Best quality, 4x (default)
    GENERAL_X2 = "RealESRGAN_x2plus"      # 2x upscaling
    ANIME_X4 = "RealESRGAN_x4plus_anime"  # Optimized for anime/illustrations
    FAST_X4 = "realesr-general-x4v3"      # Faster, good quality

class ResizeMode(Enum):
    LANCZOS = "lanczos"    # Best for downscaling
    BICUBIC = "bicubic"    # Good balance
    BILINEAR = "bilinear"  # Fast
    NEAREST = "nearest"    # Pixel-perfect

class VectorMode(Enum):
    PHOTO = "photo"              # Best for photographs
    ILLUSTRATION = "illustration" # Best for illustrations
    LOGO = "logo"                # Best for logos
    PIXEL_ART = "pixel_art"      # Best for pixel art
```

## Legacy Scripts

The original scripts are still available and functional:

| Script | Purpose |
|--------|---------|
| `optimize_images.py` | Resize PNGs to 256x256 with optimization |
| `create_banner.py` | Create Twitter banners (1500x500) |
| `create_letterbox_banner.py` | Create letterbox banners |
| `reduced_background_images.py` | Compress images to <200KB |

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI image upscaling (the magic!)
- [rembg](https://github.com/danielgatis/rembg) - AI background removal
- [vtracer](https://github.com/visioncortex/vtracer) - Raster to vector conversion
- [Pillow](https://pillow.readthedocs.io/) - Image processing
- [OpenCV](https://opencv.org/) - Computer vision
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [watchdog](https://github.com/gorakhargosh/watchdog) - Folder monitoring
