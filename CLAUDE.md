# CLAUDE.md - Claude Code Project Guide

## Project Overview

This is an AI-powered photo editor CLI. Use the `photo` command for all operations.

## Quick Commands

```bash
photo banner image.png      # Create banner (1500x500)
photo button image.png      # Create button (200x60)
photo postcard image.png    # Print postcard
photo flyer image.png       # Print flyer
photo icon image.png        # App icon (256x256)

photo upscale image.png     # AI upscale 4x
photo rembg image.png       # Remove background
photo watermark image.png              # Remove Gemini watermark (bottom-right)
photo watermark image.png bottom-left  # Different position
photo vector image.png      # Convert to SVG

photo list                  # Show all presets
```

## When User Asks To...

| User says... | Run this |
|--------------|----------|
| "make a banner" | `photo banner <image>` |
| "make a banner but keep the whole image" | `photo banner <image> --fit` |
| "create a button" | `photo button <image>` |
| "make it bigger" / "upscale" | `photo upscale <image>` |
| "remove background" | `photo rembg <image>` |
| "remove watermark" / "remove Gemini watermark" | **Install LaMa first:** `pip install simple-lama-inpainting opencv-python`, then `photo watermark <image>` |
| "remove watermark from bottom-left" | `photo watermark <image> bottom-left` |

> **IMPORTANT:** Watermark removal requires LaMa for quality results. Without it, you get blurry output. Run `pip install simple-lama-inpainting opencv-python` before using `photo watermark`. Verify with output showing `[LaMa] Inpainting model loaded`.
| "make it a vector" / "SVG" | `photo vector <image>` |
| "postcard" / "print" | `photo postcard <image>` |
| "social media sizes" | `python format_converter.py <image> --all-social` |

## Output Location

All outputs go to `./output/` folder, organized by type:
- `output/no_watermark/` - Watermark-removed images
- `output/no_background/` - Background-removed images
- `output/upscaled/` - AI-upscaled images
- `output/vectors/` - SVG files
- `output/social/` - Social media formats
- `output/print/` - Print formats
- `output/web/` - Web formats

## Key Files

- `photo` - Main CLI (use this)
- `photo_editor.py` - Core Python module
- `format_converter.py` - Batch conversion
- `presets.py` - Format definitions
- `api_server.py` - REST API

## Development Notes

- Python 3.10+ required
- Uses Apple Silicon MPS when available
- AI models download on first use (~200MB total)
- All processing is local (no API calls)
