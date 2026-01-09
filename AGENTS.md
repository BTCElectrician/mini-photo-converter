# AGENTS.md - AI Assistant Guide

> This file helps AI assistants (Claude, Codex, Cursor, GPT, etc.) understand and use this photo editing system.

## What This Project Does

This is a **command-line photo editor** with AI capabilities. It converts images to different formats (banners, buttons, postcards) and applies AI processing (upscaling, background removal, vectorization).

**Primary use case:** Processing AI-generated images (from Gemini, DALL-E, Midjourney) into usable formats for web/mobile apps.

## Quick Reference - The `photo` Command

```bash
# FORMAT CONVERSION (instant)
photo banner <image>        # Twitter/social banner (1500x500)
photo button <image>        # Web button (200x60)
photo postcard <image>      # Print postcard (1800x1200, 300dpi)
photo flyer <image>         # Print flyer (2550x3300, 300dpi)
photo icon <image>          # App icon (256x256)
photo thumbnail <image>     # Thumbnail (300x300)
photo hero <image>          # Hero image (1920x1080)
photo instagram <image>     # Instagram square (1080x1080)
photo youtube <image>       # YouTube thumbnail (1280x720)

# AI PROCESSING (requires models, slower)
photo upscale <image>       # AI upscale 4x (Real-ESRGAN)
photo upscale <image> 2     # AI upscale 2x
photo rembg <image>         # Remove background (AI)
photo vector <image>        # Convert to SVG

# UTILITIES
photo resize <image> <w> <h>  # Resize to exact dimensions
photo list                    # Show all 30+ presets
photo help                    # Show help
```

## Output Location

All outputs go to `./output/` organized by type:
```
output/
├── social/          # Social media (twitter/, instagram/, etc.)
├── print/           # Print formats (postcards/, flyers/, posters/)
├── web/             # Web formats (buttons/, icons/, thumbnails/)
├── upscaled/        # AI-upscaled images
├── no_background/   # Background-removed images
└── vectors/         # SVG files
```

## Common Tasks

### "Make this image a banner"
```bash
photo banner image.png
# Output: output/social/twitter/image_twitter_banner.png
```

### "Make this a button"
```bash
photo button logo.png
# Output: output/web/buttons/logo_button.png
```

### "This AI image is too small, make it bigger"
```bash
photo upscale small_image.png
# Output: output/upscaled/small_image_upscaled_4x.png (4x larger)
```

### "Remove the background"
```bash
photo rembg photo.jpg
# Output: output/no_background/photo_nobg.png (transparent PNG)
```

### "Convert to vector/SVG"
```bash
photo vector logo.png
# Output: output/vectors/logo.svg
```

### "Create multiple formats at once"
```bash
# Use format_converter.py for batch
python format_converter.py image.png banner button icon postcard
python format_converter.py image.png --all-social  # All social formats
python format_converter.py image.png --all-print   # All print formats
```

## All Available Presets

| Command | Size | Use Case |
|---------|------|----------|
| `banner` | 1500x500 | Twitter/X header |
| `button` | 200x60 | Web buttons |
| `button_small` | 120x36 | Small buttons |
| `button_large` | 300x90 | Large buttons |
| `icon` | 256x256 | App icons |
| `icon_64` | 64x64 | Small icons |
| `icon_128` | 128x128 | Medium icons |
| `icon_512` | 512x512 | Large icons |
| `favicon` | 32x32 | Browser favicon |
| `thumbnail` | 300x300 | Preview thumbnails |
| `hero` | 1920x1080 | Website hero images |
| `avatar` | 400x400 | Profile pictures |
| `og_image` | 1200x630 | Social share preview |
| `twitter_banner` | 1500x500 | Twitter header |
| `twitter_post` | 1200x675 | Twitter post image |
| `instagram_square` | 1080x1080 | Instagram post |
| `instagram_story` | 1080x1920 | Instagram story |
| `instagram_portrait` | 1080x1350 | Instagram portrait |
| `facebook_cover` | 820x312 | Facebook cover |
| `facebook_post` | 1200x630 | Facebook post |
| `linkedin_banner` | 1584x396 | LinkedIn header |
| `linkedin_post` | 1200x627 | LinkedIn post |
| `youtube_thumbnail` | 1280x720 | YouTube thumbnail |
| `youtube_banner` | 2560x1440 | YouTube channel art |
| `postcard` | 1800x1200 | 6x4" postcard (300dpi) |
| `postcard_5x7` | 2100x1500 | 5x7" postcard (300dpi) |
| `flyer` | 2550x3300 | Letter flyer (300dpi) |
| `flyer_a4` | 2480x3508 | A4 flyer (300dpi) |
| `poster` | 3300x5100 | 11x17" poster (300dpi) |
| `business_card` | 1050x600 | Business card (300dpi) |

## Alternative Interfaces

### Python API
```python
from photo_editor import PhotoEditor, quick_upscale, quick_remove_bg

editor = PhotoEditor()

# AI upscale
result = editor.ai_upscale("small.png", scale=4)

# Remove background
result = editor.remove_background("photo.jpg")

# Quick one-liners
quick_upscale("art.png")      # -> art_upscaled_4x.png
quick_remove_bg("photo.jpg")  # -> photo_nobg.png
```

### REST API
```bash
# Start server
python api_server.py  # Runs on http://localhost:8000

# Endpoints
POST /upscale         # AI upscale
POST /remove-bg       # Remove background
POST /vectorize       # Convert to SVG
POST /convert/{preset} # Convert to preset format
GET  /presets         # List all presets
GET  /docs            # Interactive API docs
```

### Interactive Mode
```bash
python smart_cli.py
> create a button from logo.png
> upscale my_art.png 4x
> remove background from portrait.jpg
> exit
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                         │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   photo      │  smart_cli   │  api_server  │ format_converter│
│   (simple)   │ (natural)    │   (REST)     │    (batch)      │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       └──────────────┴──────────────┴────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   photo_editor.py  │  ← Core processing
                    │   (PhotoEditor)    │
                    └─────────┬─────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
┌──────▼──────┐      ┌────────▼────────┐     ┌──────▼──────┐
│ Real-ESRGAN │      │     rembg       │     │   vtracer   │
│ (upscale)   │      │ (bg removal)    │     │ (vectorize) │
└─────────────┘      └─────────────────┘     └─────────────┘
```

## File Purposes

| File | Purpose |
|------|---------|
| `photo` | Simple CLI entry point (recommended for AI use) |
| `photo_editor.py` | Core PhotoEditor class with all AI methods |
| `format_converter.py` | Batch format conversion with presets |
| `presets.py` | Preset definitions (sizes, DPI, output folders) |
| `api_server.py` | FastAPI REST server for web/mobile apps |
| `smart_cli.py` | Natural language interactive CLI |
| `drop_watcher.py` | Folder watcher for auto-processing |
| `setup.sh` | Adds `photo` command to PATH |

## Requirements

- Python 3.10+
- ~500MB disk for AI models (downloaded on first use)
- Apple Silicon: Uses MPS GPU acceleration automatically
- NVIDIA: Uses CUDA automatically
- CPU: Works but slower for AI operations

## Installation

```bash
# Clone and setup
git clone <repo>
cd mini-photo-converter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add 'photo' to PATH
./setup.sh && source ~/.zshrc

# Verify
photo help
```

## For AI Agents: Integration Patterns

### Pattern 1: Direct CLI calls
```bash
# Most reliable - just call the photo command
photo banner user_uploaded_image.png
```

### Pattern 2: Python import
```python
# For more control
from photo_editor import PhotoEditor
editor = PhotoEditor()
result = editor.ai_upscale(input_path, output_path, scale=4)
if result.success:
    print(f"Output: {result.output_path}")
```

### Pattern 3: REST API (for web apps)
```javascript
const formData = new FormData();
formData.append('file', imageFile);
const response = await fetch('http://localhost:8000/convert/banner', {
  method: 'POST',
  body: formData
});
const blob = await response.blob();
```

## Error Handling

The `photo` command returns exit codes:
- `0` = Success
- `1` = Error (file not found, invalid command, processing failed)

Check output messages for details:
```bash
photo banner nonexistent.png
# Error: File not found: nonexistent.png
# Exit code: 1
```

## Notes for AI Assistants

1. **Always use absolute paths** when possible to avoid working directory issues
2. **Check file exists** before processing: `test -f image.png && photo banner image.png`
3. **Output is predictable**: `output/<category>/<type>/filename_suffix.ext`
4. **AI operations download models on first use** (~170MB for rembg, ~64MB per Real-ESRGAN model)
5. **For batch operations**, use `format_converter.py` instead of multiple `photo` calls
6. **The `photo` command is stateless** - each call is independent
