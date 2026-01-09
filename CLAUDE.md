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
photo vector image.png      # Convert to SVG

photo list                  # Show all presets
```

## When User Asks To...

| User says... | Run this |
|--------------|----------|
| "make a banner" | `photo banner <image>` |
| "create a button" | `photo button <image>` |
| "make it bigger" / "upscale" | `photo upscale <image>` |
| "remove background" | `photo rembg <image>` |
| "make it a vector" / "SVG" | `photo vector <image>` |
| "postcard" / "print" | `photo postcard <image>` |
| "social media sizes" | `python format_converter.py <image> --all-social` |

## Output Location

All outputs go to `./output/` folder, organized by type.

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
