# Mini Photo Converter

A collection of Python scripts for image processing and optimization, designed to handle various image conversion tasks efficiently.

## Features

### 1. Image Optimization (`optimize_images.py`)
- Resizes PNG images to 256x256 pixels
- Applies high-quality optimization (95% quality)
- Organizes processed images into separate directories
- Maintains original image quality while reducing file size

### 2. Banner Creation (`create_banner.py`)
- Creates banner images from source photos
- Processes images for banner display
- Saves processed banners in a dedicated directory

### 3. Letterbox Banner Creation (`create_letterbox_banner.py`)
- Generates letterbox-style banner images
- Maintains aspect ratio while fitting to banner dimensions
- Processes and organizes letterbox banners

### 4. Background Image Reduction (`reduced_background_images.py`)
- Reduces background image sizes
- Optimizes images for background use
- Maintains quality while reducing file size

## Prerequisites

- Python 3.x
- Pillow (PIL Fork)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mini-photo-converter.git
cd mini-photo-converter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Image Optimization
```bash
python optimize_images.py
```
- Place PNG images in the script directory
- Optimized images will be saved in `optimized-hq/`
- Original images will be moved to `processed-photos/`

### Banner Creation
```bash
python create_banner.py
```
- Source images will be processed into banners
- Output will be saved in `processed-banner-photos/`

### Letterbox Banner Creation
```bash
python create_letterbox_banner.py
```
- Creates letterbox-style banners
- Output will be saved in `letterbox-banners/`

### Background Image Reduction
```bash
python reduced_background_images.py
```
- Reduces background image sizes
- Output will be saved in `reduced_background_images/`

## Directory Structure

```
mini-photo-converter/
├── optimized-hq/           # High-quality optimized images
├── processed-photos/       # Original processed images
├── processed-banner-photos/# Processed banner images
├── letterbox-banners/      # Letterbox-style banner images
├── reduced_background_images/ # Reduced background images
├── optimize_images.py      # Image optimization script
├── create_banner.py        # Banner creation script
├── create_letterbox_banner.py # Letterbox banner creation script
├── reduced_background_images.py # Background image reduction script
└── requirements.txt        # Project dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Pillow (PIL Fork) for image processing capabilities
- Python community for excellent documentation and support 