# SuperCharged Photo Editor

A modern, AI-powered photo editing toolkit for web and mobile apps. Process AI-generated images with background removal, vectorization, and perfect resizing - all in one place.

## New Features

### AI-Powered Background Removal
Remove backgrounds instantly using deep learning (U2-Net model via rembg). Perfect for product photos, AI-generated art, and graphics.

### Raster to Vector Conversion
Convert any image to scalable SVG using vtracer. Multiple presets for photos, illustrations, logos, and pixel art.

### Smart Resizing
High-quality resizing with multiple algorithms (LANCZOS, BICUBIC, BILINEAR) and optional OpenCV super-resolution.

### Drop Folder Watcher
Drop images into a folder and watch them get processed automatically. Perfect for batch workflows.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a single image (full pipeline)
python photo_editor.py image.png --pipeline

# Remove background only
python photo_editor.py image.png --remove-bg

# Convert to vector SVG
python photo_editor.py image.png --vectorize

# Resize to specific dimensions
python photo_editor.py image.png --resize 1024 1024

# Start the drop folder watcher
python drop_watcher.py
```

## Usage Examples

### Python API

```python
from photo_editor import PhotoEditor, VectorMode, ResizeMode

editor = PhotoEditor()

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
from photo_editor import quick_remove_bg, quick_vectorize, quick_resize

# One-liners for common operations
quick_remove_bg("photo.png")           # -> photo_nobg.png
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
| **rembg** | AI background removal using U2-Net deep learning |
| **vtracer** | Fast raster-to-vector conversion (Rust-based) |
| **Pillow** | Core image processing |
| **OpenCV** | Advanced resizing algorithms |
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

```bash
# Clone the repository
git clone https://github.com/yourusername/mini-photo-converter.git
cd mini-photo-converter

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Run Note
The first time you use background removal, rembg will download the U2-Net model (~170MB). This is automatic and only happens once.

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

    def remove_background(self, input_path, output_path=None, alpha_matting=False)
    def vectorize(self, input_path, output_path=None, mode=VectorMode.ILLUSTRATION)
    def smart_resize(self, input_path, output_path=None, width=None, height=None,
                     scale=None, mode=ResizeMode.LANCZOS)
    def process_full_pipeline(self, input_path, output_dir=None, ...)
    def batch_process(self, input_dir, output_dir=None, ...)
```

### Enums

```python
class ResizeMode(Enum):
    LANCZOS = "lanczos"    # Best for downscaling
    BICUBIC = "bicubic"    # Good balance
    BILINEAR = "bilinear"  # Fast
    NEAREST = "nearest"    # Pixel-perfect
    SUPERRES = "superres"  # OpenCV super-resolution

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

- [rembg](https://github.com/danielgatis/rembg) - AI background removal
- [vtracer](https://github.com/nicmcd/vtracer) - Raster to vector conversion
- [Pillow](https://pillow.readthedocs.io/) - Image processing
- [OpenCV](https://opencv.org/) - Computer vision
- [watchdog](https://github.com/gorakhargosh/watchdog) - Folder monitoring
