# 📸 Photo Editor User Guide

Welcome to the **SuperCharged Photo Editor** — your all-in-one toolkit for transforming AI-generated images into professional formats.

---

## 🚀 Quick Start

### 1. Install
```bash
cd mini-photo-converter
pip install -r requirements.txt
./setup.sh
source ~/.zshrc
```

### 2. Use
```bash
photo banner my_image.png
```

That's it! Your banner is now in `output/social/twitter/`

---

## 🎨 What Can You Do?

### Create Social Media Graphics
| Command | What You Get |
|---------|--------------|
| `photo banner image.png` | Twitter/X header (1500×500) |
| `photo instagram image.png` | Instagram post (1080×1080) |
| `photo story image.png` | Instagram story (1080×1920) |
| `photo youtube image.png` | YouTube thumbnail (1280×720) |
| `photo facebook image.png` | Facebook post (1200×630) |
| `photo linkedin image.png` | LinkedIn post (1200×627) |

### Create Print Materials
| Command | What You Get |
|---------|--------------|
| `photo postcard image.png` | 6×4" postcard (300 DPI) |
| `photo flyer image.png` | Letter size flyer (300 DPI) |
| `photo poster image.png` | 11×17" poster (300 DPI) |
| `photo card image.png` | Business card (300 DPI) |

### Create Web Assets
| Command | What You Get |
|---------|--------------|
| `photo button image.png` | Web button (200×60) |
| `photo icon image.png` | App icon (256×256) |
| `photo favicon image.png` | Browser favicon (32×32) |
| `photo thumbnail image.png` | Thumbnail (300×300) |
| `photo hero image.png` | Hero image (1920×1080) |

---

## ✨ AI Superpowers

### 🔍 Upscale Small Images
Got a small AI-generated image? Make it **4× bigger** without losing quality!

```bash
photo upscale my_art.png
```

**Before:** 512×512 → **After:** 2048×2048

The AI actually *adds* detail instead of blur. Magic! ✨

```bash
# 2× upscale (less aggressive)
photo upscale my_art.png 2
```

### 🪄 Remove Backgrounds
Cut out subjects instantly with AI:

```bash
photo rembg portrait.jpg
```

**Result:** Transparent PNG with just the subject. Perfect for:
- Product photos
- Profile pictures
- Stickers & graphics

### 🧹 Remove Watermarks
Got a Gemini watermark? Remove it with AI inpainting:

```bash
photo watermark gemini_art.png
```

The AI fills in the watermark area seamlessly. Works with:
- Gemini watermarks (bottom-right)
- Other watermarks (specify position)

```bash
# Different positions
photo watermark image.png bottom-left
photo watermark image.png top-right
```

### 🎯 Convert to Vector (SVG)
Turn any image into scalable vector graphics:

```bash
photo vector logo.png
```

Great for logos, icons, and illustrations that need to scale infinitely.

---

## 📁 Where Do Files Go?

All outputs are organized automatically:

```
output/
├── social/
│   ├── twitter/          ← Banners & posts
│   ├── instagram/        ← Posts & stories
│   ├── facebook/         ← Covers & posts
│   └── youtube/          ← Thumbnails & banners
│
├── print/
│   ├── postcards/        ← Postcards
│   ├── flyers/           ← Flyers
│   └── posters/          ← Posters
│
├── web/
│   ├── buttons/          ← Buttons
│   ├── icons/            ← Icons & favicons
│   └── thumbnails/       ← Thumbnails
│
├── upscaled/             ← AI-upscaled images
├── no_background/        ← Background-removed images
├── no_watermark/         ← Watermark-removed images
└── vectors/              ← SVG files
```

---

## 🎯 Common Workflows

### Workflow 1: AI Art → Social Media
```bash
# 1. Remove the Gemini watermark
photo watermark gemini_art.png

# 2. Upscale to high resolution
photo upscale output/no_watermark/gemini_art_nowm.png

# 3. Create your social posts
photo banner output/upscaled/gemini_art_nowm_upscaled_4x.png
photo instagram output/upscaled/gemini_art_nowm_upscaled_4x.png
```

### Workflow 2: Logo → All Sizes
```bash
# Create all web sizes at once
photo icon logo.png
photo favicon logo.png
photo thumbnail logo.png

# Convert to vector for infinite scaling
photo vector logo.png
```

### Workflow 3: Photo → Print Ready
```bash
# Remove background first
photo rembg product_photo.jpg

# Create print materials
photo postcard output/no_background/product_photo_nobg.png
photo flyer output/no_background/product_photo_nobg.png
```

---

## 📋 All Commands Reference

```bash
# FORMAT CONVERSION
photo banner <image>        # Twitter banner (1500×500)
photo button <image>        # Web button (200×60)
photo icon <image>          # App icon (256×256)
photo favicon <image>       # Favicon (32×32)
photo thumbnail <image>     # Thumbnail (300×300)
photo hero <image>          # Hero image (1920×1080)
photo avatar <image>        # Profile pic (400×400)

photo instagram <image>     # Instagram post (1080×1080)
photo story <image>         # Instagram story (1080×1920)
photo youtube <image>       # YouTube thumbnail (1280×720)
photo facebook <image>      # Facebook post (1200×630)
photo linkedin <image>      # LinkedIn post (1200×627)

photo postcard <image>      # Postcard 6×4" (300 DPI)
photo flyer <image>         # Letter flyer (300 DPI)
photo poster <image>        # Poster 11×17" (300 DPI)
photo card <image>          # Business card (300 DPI)

# AI PROCESSING
photo upscale <image>       # Upscale 4× with AI
photo upscale <image> 2     # Upscale 2× with AI
photo rembg <image>         # Remove background
photo watermark <image>     # Remove watermark
photo vector <image>        # Convert to SVG

# UTILITIES
photo resize <image> W H    # Resize to exact dimensions
photo list                  # Show all 30+ presets
photo help                  # Show help
```

---

## 💡 Pro Tips

### Tip 1: Chain Commands
Process images step by step for best results:
```bash
photo watermark art.png && photo upscale output/no_watermark/art_nowm.png
```

### Tip 2: Batch Processing
Use the format converter for multiple formats at once:
```bash
python format_converter.py image.png banner instagram youtube postcard
```

Or all social sizes:
```bash
python format_converter.py image.png --all-social
```

### Tip 3: Upscale Before Converting
For best quality, upscale your AI art first, then convert to formats:
```bash
photo upscale small_art.png
photo banner output/upscaled/small_art_upscaled_4x.png
```

### Tip 4: Use Aliases
Quick shortcuts for faster typing:
- `btn` → button
- `ig` → instagram
- `yt` → youtube
- `fb` → facebook
- `thumb` → thumbnail

---

## ❓ Troubleshooting

### "Command not found: photo"
Run the setup script and reload your shell:
```bash
./setup.sh
source ~/.zshrc   # or ~/.bashrc
```

### "Models downloading..."
First time using AI features? Models download automatically:
- **Real-ESRGAN:** ~64MB (upscaling)
- **U2-Net:** ~170MB (background removal)
- **LaMa:** ~200MB (watermark removal)

This only happens once. They're cached for future use.

### "Out of memory"
For very large images, try upscaling 2× instead of 4×:
```bash
photo upscale large_image.png 2
```

### Slow performance?
- Close other GPU-intensive apps
- Apple Silicon users: Make sure MPS is working (you should see `[Real-ESRGAN] Using device: mps`)

---

## 🙋 Need Help?

- **List all presets:** `photo list`
- **Command help:** `photo help`
- **Full documentation:** See `AGENTS.md`

---

Made with ❤️ for creators who love AI-generated art.
