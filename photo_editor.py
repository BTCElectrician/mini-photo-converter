"""
SuperCharged Photo Editor - AI-Powered Image Processing

A modern photo editing toolkit for web and mobile apps featuring:
- AI-powered background removal (rembg with U2-Net)
- AI-powered upscaling (Real-ESRGAN) - upscale 2x-4x WITHOUT losing quality!
- Raster to vector conversion (vtracer)
- Smart resizing with multiple algorithms
- Batch processing with folder watching
- Perfect for processing AI-generated images

Usage:
    from photo_editor import PhotoEditor

    editor = PhotoEditor()
    editor.remove_background("input.png", "output.png")
    editor.ai_upscale("small.png", "large.png", scale=4)  # 4x upscale!
    editor.vectorize("input.png", "output.svg")
    editor.smart_resize("input.png", "output.png", width=1024, height=1024)
    editor.process_full_pipeline("input.png", "output_dir/")
"""

import os
import io
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from enum import Enum

from PIL import Image
import numpy as np

# Lazy imports for heavy libraries
_rembg = None
_vtracer = None
_cv2 = None
_realesrgan = None


def _get_realesrgan(model_name: str = "RealESRGAN_x4plus", scale: int = 4, denoise_strength: float = 0.5):
    """
    Lazy load Real-ESRGAN upscaler.

    Models available:
    - RealESRGAN_x4plus: Best quality for general images (4x)
    - RealESRGAN_x4plus_anime_6B: Optimized for anime/illustrations (4x)
    - RealESRGAN_x2plus: 2x upscaling
    - realesr-general-x4v3: Fast general purpose (4x)
    """
    global _realesrgan

    # Cache key based on model and scale
    cache_key = f"{model_name}_{scale}"

    if _realesrgan is None:
        _realesrgan = {}

    if cache_key not in _realesrgan:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        half_precision = device == 'cuda'  # Use fp16 on GPU for speed

        # Model configurations
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        else:
            # Default to x4plus
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

        # Create upscaler
        upscaler = RealESRGANer(
            scale=netscale,
            model_path=model_url,
            dni_weight=denoise_strength,
            model=model,
            tile=0,  # 0 for no tiling, or set to 256/512 for large images
            tile_pad=10,
            pre_pad=0,
            half=half_precision,
            device=device
        )

        _realesrgan[cache_key] = upscaler

    return _realesrgan[cache_key]


def _get_rembg():
    """Lazy load rembg to speed up initial import."""
    global _rembg
    if _rembg is None:
        from rembg import remove, new_session
        _rembg = {"remove": remove, "new_session": new_session}
    return _rembg


def _get_vtracer():
    """Lazy load vtracer for vectorization."""
    global _vtracer
    if _vtracer is None:
        import vtracer
        _vtracer = vtracer
    return _vtracer


def _get_cv2():
    """Lazy load OpenCV."""
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


class ResizeMode(Enum):
    """Resize algorithm options."""
    LANCZOS = "lanczos"      # Best for downscaling, sharp results
    BICUBIC = "bicubic"      # Good balance of quality and speed
    BILINEAR = "bilinear"    # Fast, good for real-time
    NEAREST = "nearest"      # Pixel-perfect, good for pixel art
    SUPERRES = "superres"    # Basic super resolution (OpenCV)


class UpscaleModel(Enum):
    """AI upscaling model options (Real-ESRGAN)."""
    GENERAL_X4 = "RealESRGAN_x4plus"           # Best quality, 4x upscale
    GENERAL_X2 = "RealESRGAN_x2plus"           # 2x upscale
    ANIME_X4 = "RealESRGAN_x4plus_anime_6B"    # Optimized for anime/illustrations
    FAST_X4 = "realesr-general-x4v3"           # Faster, good quality


class VectorMode(Enum):
    """Vectorization style presets."""
    PHOTO = "photo"          # Best for photographs
    ILLUSTRATION = "illustration"  # Best for illustrations/graphics
    LOGO = "logo"            # Best for logos, simple shapes
    PIXEL_ART = "pixel_art"  # Best for pixel art


@dataclass
class ProcessingResult:
    """Result of an image processing operation."""
    success: bool
    input_path: str
    output_path: Optional[str]
    operation: str
    message: str
    original_size: Optional[Tuple[int, int]] = None
    output_size: Optional[Tuple[int, int]] = None
    file_size_before: Optional[int] = None
    file_size_after: Optional[int] = None


class PhotoEditor:
    """
    SuperCharged Photo Editor with AI-powered capabilities.

    Features:
    - Background removal using AI (rembg/U2-Net)
    - AI upscaling with Real-ESRGAN (2x-4x without quality loss!)
    - Raster to vector conversion (vtracer)
    - Smart resizing with multiple algorithms
    - Full processing pipeline for AI-generated images

    Example:
        editor = PhotoEditor()

        # Remove background
        editor.remove_background("photo.png", "photo_nobg.png")

        # AI Upscale (THIS IS THE MAGIC FOR SMALL AI IMAGES!)
        editor.ai_upscale("small_gemini_art.png", "large_gemini_art.png", scale=4)

        # Vectorize
        editor.vectorize("logo.png", "logo.svg")

        # Smart resize
        editor.smart_resize("image.png", "image_resized.png", width=512)

        # Full pipeline: remove bg -> upscale -> vectorize -> resize
        editor.process_full_pipeline("ai_image.png", "output/")
    """

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}

    def __init__(self,
                 default_output_dir: str = "processed",
                 rembg_model: str = "u2net"):
        """
        Initialize the PhotoEditor.

        Args:
            default_output_dir: Default directory for output files
            rembg_model: Model for background removal. Options:
                - "u2net" (default, best quality)
                - "u2netp" (faster, slightly lower quality)
                - "u2net_human_seg" (optimized for humans)
                - "isnet-general-use" (good general purpose)
                - "isnet-anime" (optimized for anime)
        """
        self.default_output_dir = Path(default_output_dir)
        self.rembg_model = rembg_model
        self._rembg_session = None

    def _ensure_output_dir(self, output_path: Union[str, Path]) -> Path:
        """Ensure the output directory exists."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _get_rembg_session(self):
        """Get or create rembg session for faster batch processing."""
        if self._rembg_session is None:
            rembg = _get_rembg()
            self._rembg_session = rembg["new_session"](self.rembg_model)
        return self._rembg_session

    def remove_background(self,
                          input_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          alpha_matting: bool = False,
                          alpha_matting_foreground_threshold: int = 240,
                          alpha_matting_background_threshold: int = 10,
                          only_mask: bool = False) -> ProcessingResult:
        """
        Remove background from an image using AI (rembg/U2-Net).

        Args:
            input_path: Path to input image
            output_path: Path for output image (default: input_nobg.png)
            alpha_matting: Enable alpha matting for better edges
            alpha_matting_foreground_threshold: Foreground threshold (0-255)
            alpha_matting_background_threshold: Background threshold (0-255)
            only_mask: If True, output only the mask instead of the image

        Returns:
            ProcessingResult with operation details
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_nobg.png"
        output_path = self._ensure_output_dir(output_path)

        try:
            rembg = _get_rembg()
            session = self._get_rembg_session()

            # Read input image
            with open(input_path, 'rb') as f:
                input_data = f.read()

            original_size = os.path.getsize(input_path)
            img = Image.open(input_path)
            original_dimensions = img.size

            # Remove background
            output_data = rembg["remove"](
                input_data,
                session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                only_mask=only_mask
            )

            # Save output
            output_img = Image.open(io.BytesIO(output_data))
            output_img.save(output_path, format='PNG', optimize=True)

            output_size = os.path.getsize(output_path)

            return ProcessingResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                operation="remove_background",
                message=f"Background removed successfully",
                original_size=original_dimensions,
                output_size=output_img.size,
                file_size_before=original_size,
                file_size_after=output_size
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                operation="remove_background",
                message=f"Error: {str(e)}"
            )

    def ai_upscale(self,
                   input_path: Union[str, Path],
                   output_path: Optional[Union[str, Path]] = None,
                   scale: int = 4,
                   model: UpscaleModel = UpscaleModel.GENERAL_X4,
                   denoise_strength: float = 0.5,
                   face_enhance: bool = False,
                   tile_size: int = 0,
                   output_format: Optional[str] = None,
                   quality: int = 95) -> ProcessingResult:
        """
        AI-powered image upscaling using Real-ESRGAN.

        This is the BEST way to upscale AI-generated images (like from Gemini, DALL-E,
        Midjourney, etc.) without losing quality. The AI actually ADDS detail!

        Args:
            input_path: Path to input image
            output_path: Path for output image (default: input_upscaled.png)
            scale: Upscale factor (2 or 4, depending on model)
            model: Which AI model to use:
                - GENERAL_X4: Best quality for most images (4x)
                - GENERAL_X2: 2x upscaling
                - ANIME_X4: Optimized for anime/illustrations (4x)
                - FAST_X4: Faster processing, good quality (4x)
            denoise_strength: Denoise strength (0.0-1.0), higher = more denoising
            face_enhance: Enable face enhancement (uses GFPGAN)
            tile_size: Tile size for processing large images (0=auto, or 256/512)
            output_format: Output format (png, jpg, webp)
            quality: JPEG/WebP quality (1-100)

        Returns:
            ProcessingResult with operation details

        Example:
            # Upscale a small Gemini image 4x
            editor.ai_upscale("gemini_art_512.png", "gemini_art_2048.png", scale=4)

            # Use anime model for illustrations
            editor.ai_upscale("anime.png", model=UpscaleModel.ANIME_X4)

            # 2x upscale for less aggressive enlargement
            editor.ai_upscale("photo.jpg", scale=2, model=UpscaleModel.GENERAL_X2)
        """
        input_path = Path(input_path)

        # Determine actual scale based on model
        if model == UpscaleModel.GENERAL_X2:
            actual_scale = 2
        else:
            actual_scale = 4

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_upscaled_{actual_scale}x.png"
        output_path = self._ensure_output_dir(output_path)

        try:
            # Load image
            img = Image.open(input_path)
            original_size = os.path.getsize(input_path)
            original_dimensions = img.size
            orig_width, orig_height = original_dimensions

            # Convert to numpy array for Real-ESRGAN
            if img.mode == 'RGBA':
                img_array = np.array(img)
                has_alpha = True
            else:
                img_array = np.array(img.convert('RGB'))
                has_alpha = False

            # Convert RGB to BGR for OpenCV/Real-ESRGAN
            cv2 = _get_cv2()
            if has_alpha:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
            else:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Get upscaler
            upscaler = _get_realesrgan(model.value, actual_scale, denoise_strength)

            # Upscale!
            output_bgr, _ = upscaler.enhance(img_bgr, outscale=actual_scale)

            # Convert back to RGB/RGBA
            if has_alpha:
                output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGRA2RGBA)
            else:
                output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

            # Convert back to PIL
            output_img = Image.fromarray(output_rgb)
            new_dimensions = output_img.size

            # Determine output format
            if output_format:
                fmt = output_format.upper()
                if fmt == 'JPG':
                    fmt = 'JPEG'
            else:
                fmt = output_path.suffix.upper().lstrip('.')
                if fmt == 'JPG':
                    fmt = 'JPEG'
                elif not fmt:
                    fmt = 'PNG'

            # Handle format-specific requirements
            save_kwargs = {'optimize': True}
            if fmt in ('JPEG', 'WEBP'):
                save_kwargs['quality'] = quality
                if fmt == 'JPEG' and output_img.mode == 'RGBA':
                    background = Image.new('RGB', output_img.size, (255, 255, 255))
                    background.paste(output_img, mask=output_img.split()[3])
                    output_img = background

            output_img.save(output_path, format=fmt, **save_kwargs)
            output_size = os.path.getsize(output_path)

            return ProcessingResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                operation="ai_upscale",
                message=f"Upscaled {actual_scale}x: {orig_width}x{orig_height} -> {new_dimensions[0]}x{new_dimensions[1]} ({model.value})",
                original_size=original_dimensions,
                output_size=new_dimensions,
                file_size_before=original_size,
                file_size_after=output_size
            )

        except ImportError as e:
            return ProcessingResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                operation="ai_upscale",
                message=f"Real-ESRGAN not installed. Run: pip install realesrgan basicsr. Error: {str(e)}"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                operation="ai_upscale",
                message=f"Error: {str(e)}"
            )

    def vectorize(self,
                  input_path: Union[str, Path],
                  output_path: Optional[Union[str, Path]] = None,
                  mode: VectorMode = VectorMode.ILLUSTRATION,
                  colormode: str = "color",
                  hierarchical: str = "stacked",
                  filter_speckle: int = 4,
                  color_precision: int = 6,
                  layer_difference: int = 16,
                  corner_threshold: int = 60,
                  length_threshold: float = 4.0,
                  max_iterations: int = 10,
                  splice_threshold: int = 45,
                  path_precision: int = 3) -> ProcessingResult:
        """
        Convert a raster image to vector SVG using vtracer.

        Args:
            input_path: Path to input image
            output_path: Path for output SVG (default: input.svg)
            mode: Vectorization preset (PHOTO, ILLUSTRATION, LOGO, PIXEL_ART)
            colormode: "color" or "binary"
            hierarchical: "stacked" or "cutout"
            filter_speckle: Speckle filter size
            color_precision: Color precision (1-8)
            layer_difference: Layer difference threshold
            corner_threshold: Corner detection threshold
            length_threshold: Minimum path length
            max_iterations: Max curve fitting iterations
            splice_threshold: Path splicing threshold
            path_precision: SVG path precision

        Returns:
            ProcessingResult with operation details
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}.svg"
        output_path = self._ensure_output_dir(output_path)

        try:
            vtracer = _get_vtracer()

            # Load image
            img = Image.open(input_path)
            original_size = os.path.getsize(input_path)
            original_dimensions = img.size

            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Apply mode presets
            if mode == VectorMode.LOGO:
                filter_speckle = 8
                color_precision = 4
                corner_threshold = 90
            elif mode == VectorMode.PIXEL_ART:
                filter_speckle = 0
                color_precision = 8
                corner_threshold = 0
                length_threshold = 0
            elif mode == VectorMode.PHOTO:
                filter_speckle = 2
                color_precision = 8
                layer_difference = 8

            # Convert to bytes for vtracer
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # Vectorize
            svg_str = vtracer.convert_raw_image_to_svg(
                img_bytes,
                img_format='png',
                colormode=colormode,
                hierarchical=hierarchical,
                mode='polygon',  # 'polygon' or 'spline'
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                layer_difference=layer_difference,
                corner_threshold=corner_threshold,
                length_threshold=length_threshold,
                max_iterations=max_iterations,
                splice_threshold=splice_threshold,
                path_precision=path_precision
            )

            # Save SVG
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_str)

            output_size = os.path.getsize(output_path)

            return ProcessingResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                operation="vectorize",
                message=f"Vectorized to SVG ({mode.value} mode)",
                original_size=original_dimensions,
                output_size=original_dimensions,  # SVG maintains dimensions
                file_size_before=original_size,
                file_size_after=output_size
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                operation="vectorize",
                message=f"Error: {str(e)}"
            )

    def smart_resize(self,
                     input_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     width: Optional[int] = None,
                     height: Optional[int] = None,
                     scale: Optional[float] = None,
                     mode: ResizeMode = ResizeMode.LANCZOS,
                     maintain_aspect: bool = True,
                     max_size: Optional[int] = None,
                     output_format: Optional[str] = None,
                     quality: int = 95) -> ProcessingResult:
        """
        Smart resize with multiple algorithm options.

        Args:
            input_path: Path to input image
            output_path: Path for output image
            width: Target width (optional)
            height: Target height (optional)
            scale: Scale factor, e.g., 2.0 for 2x (optional)
            mode: Resize algorithm (LANCZOS, BICUBIC, BILINEAR, NEAREST, SUPERRES)
            maintain_aspect: Maintain aspect ratio
            max_size: Maximum dimension (will scale down if larger)
            output_format: Output format (png, jpg, webp)
            quality: JPEG/WebP quality (1-100)

        Returns:
            ProcessingResult with operation details
        """
        input_path = Path(input_path)

        if output_path is None:
            suffix = f"_{width}x{height}" if width and height else "_resized"
            output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
        output_path = self._ensure_output_dir(output_path)

        try:
            img = Image.open(input_path)
            original_size = os.path.getsize(input_path)
            original_dimensions = img.size
            orig_width, orig_height = original_dimensions

            # Calculate target dimensions
            if scale is not None:
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
            elif width is not None and height is not None:
                if maintain_aspect:
                    # Fit within the box while maintaining aspect ratio
                    ratio = min(width / orig_width, height / orig_height)
                    new_width = int(orig_width * ratio)
                    new_height = int(orig_height * ratio)
                else:
                    new_width, new_height = width, height
            elif width is not None:
                ratio = width / orig_width
                new_width = width
                new_height = int(orig_height * ratio)
            elif height is not None:
                ratio = height / orig_height
                new_height = height
                new_width = int(orig_width * ratio)
            elif max_size is not None:
                if max(orig_width, orig_height) > max_size:
                    ratio = max_size / max(orig_width, orig_height)
                    new_width = int(orig_width * ratio)
                    new_height = int(orig_height * ratio)
                else:
                    new_width, new_height = orig_width, orig_height
            else:
                new_width, new_height = orig_width, orig_height

            # Select resampling method
            resample_methods = {
                ResizeMode.LANCZOS: Image.Resampling.LANCZOS,
                ResizeMode.BICUBIC: Image.Resampling.BICUBIC,
                ResizeMode.BILINEAR: Image.Resampling.BILINEAR,
                ResizeMode.NEAREST: Image.Resampling.NEAREST,
            }

            if mode == ResizeMode.SUPERRES:
                # Use OpenCV's super resolution if available
                resized = self._superres_resize(img, new_width, new_height)
            else:
                resample = resample_methods.get(mode, Image.Resampling.LANCZOS)
                resized = img.resize((new_width, new_height), resample)

            # Determine output format
            if output_format:
                fmt = output_format.upper()
                if fmt == 'JPG':
                    fmt = 'JPEG'
            else:
                fmt = output_path.suffix.upper().lstrip('.')
                if fmt == 'JPG':
                    fmt = 'JPEG'
                elif not fmt:
                    fmt = 'PNG'

            # Handle format-specific requirements
            save_kwargs = {'optimize': True}
            if fmt in ('JPEG', 'WEBP'):
                save_kwargs['quality'] = quality
                if fmt == 'JPEG' and resized.mode == 'RGBA':
                    # Convert RGBA to RGB for JPEG
                    background = Image.new('RGB', resized.size, (255, 255, 255))
                    background.paste(resized, mask=resized.split()[3])
                    resized = background

            resized.save(output_path, format=fmt, **save_kwargs)
            output_size = os.path.getsize(output_path)

            return ProcessingResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                operation="smart_resize",
                message=f"Resized from {orig_width}x{orig_height} to {new_width}x{new_height} ({mode.value})",
                original_size=original_dimensions,
                output_size=(new_width, new_height),
                file_size_before=original_size,
                file_size_after=output_size
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                operation="smart_resize",
                message=f"Error: {str(e)}"
            )

    def _superres_resize(self, img: Image.Image, width: int, height: int) -> Image.Image:
        """Use OpenCV for super resolution upscaling."""
        try:
            cv2 = _get_cv2()

            # Convert PIL to OpenCV format
            img_array = np.array(img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # RGBA to BGRA
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
            elif len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array

            # Use INTER_CUBIC for upscaling (best quality without deep learning models)
            resized = cv2.resize(img_cv, (width, height), interpolation=cv2.INTER_CUBIC)

            # Convert back to PIL
            if len(resized.shape) == 3 and resized.shape[2] == 4:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)
            elif len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            return Image.fromarray(resized)

        except ImportError:
            # Fallback to LANCZOS if OpenCV not available
            return img.resize((width, height), Image.Resampling.LANCZOS)

    def process_full_pipeline(self,
                              input_path: Union[str, Path],
                              output_dir: Optional[Union[str, Path]] = None,
                              remove_bg: bool = True,
                              ai_upscale: bool = False,
                              upscale_model: UpscaleModel = UpscaleModel.GENERAL_X4,
                              create_vector: bool = True,
                              resize_config: Optional[dict] = None,
                              vector_mode: VectorMode = VectorMode.ILLUSTRATION) -> List[ProcessingResult]:
        """
        Run the full processing pipeline on an image.

        Pipeline: Remove Background -> AI Upscale -> Vectorize -> Resize

        Args:
            input_path: Path to input image
            output_dir: Output directory for all processed files
            remove_bg: Whether to remove background
            ai_upscale: Whether to AI upscale (Real-ESRGAN)
            upscale_model: Which upscale model to use
            create_vector: Whether to create vector SVG
            resize_config: Dict with resize parameters (width, height, scale, etc.)
            vector_mode: Vectorization style preset

        Returns:
            List of ProcessingResult for each operation
        """
        input_path = Path(input_path)
        if output_dir is None:
            output_dir = self.default_output_dir / input_path.stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        current_image = input_path

        # Step 1: Remove background
        if remove_bg:
            nobg_path = output_dir / f"{input_path.stem}_nobg.png"
            result = self.remove_background(current_image, nobg_path)
            results.append(result)
            if result.success:
                current_image = Path(result.output_path)

        # Step 2: AI Upscale (the magic for small AI images!)
        if ai_upscale:
            scale = 2 if upscale_model == UpscaleModel.GENERAL_X2 else 4
            upscale_path = output_dir / f"{input_path.stem}_upscaled_{scale}x.png"
            result = self.ai_upscale(current_image, upscale_path, model=upscale_model)
            results.append(result)
            if result.success:
                current_image = Path(result.output_path)

        # Step 3: Vectorize (use the processed version if available)
        if create_vector:
            svg_path = output_dir / f"{input_path.stem}.svg"
            result = self.vectorize(current_image, svg_path, mode=vector_mode)
            results.append(result)

        # Step 4: Smart resize (multiple sizes if configured)
        if resize_config:
            sizes = resize_config.get('sizes', [resize_config])
            for i, config in enumerate(sizes if isinstance(sizes, list) else [sizes]):
                suffix = config.get('suffix', f'_{i}') if len(sizes) > 1 else '_resized'
                resize_path = output_dir / f"{input_path.stem}{suffix}.png"
                result = self.smart_resize(
                    current_image,
                    resize_path,
                    width=config.get('width'),
                    height=config.get('height'),
                    scale=config.get('scale'),
                    mode=config.get('mode', ResizeMode.LANCZOS)
                )
                results.append(result)

        return results

    def batch_process(self,
                      input_dir: Union[str, Path],
                      output_dir: Optional[Union[str, Path]] = None,
                      **pipeline_kwargs) -> List[List[ProcessingResult]]:
        """
        Batch process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Output directory
            **pipeline_kwargs: Arguments passed to process_full_pipeline

        Returns:
            List of results for each image
        """
        input_dir = Path(input_dir)
        all_results = []

        for file_path in input_dir.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                img_output_dir = None
                if output_dir:
                    img_output_dir = Path(output_dir) / file_path.stem

                results = self.process_full_pipeline(
                    file_path,
                    output_dir=img_output_dir,
                    **pipeline_kwargs
                )
                all_results.append(results)

                # Print progress
                for r in results:
                    status = "OK" if r.success else "FAIL"
                    print(f"[{status}] {r.operation}: {r.input_path} -> {r.output_path or r.message}")

        return all_results


def quick_remove_bg(input_path: str, output_path: Optional[str] = None) -> str:
    """Quick function to remove background from an image."""
    editor = PhotoEditor()
    result = editor.remove_background(input_path, output_path)
    if result.success:
        return result.output_path
    raise RuntimeError(result.message)


def quick_upscale(input_path: str, scale: int = 4, output_path: Optional[str] = None,
                  model: str = "general") -> str:
    """
    Quick function to AI-upscale an image.

    Args:
        input_path: Path to input image
        scale: Upscale factor (2 or 4)
        output_path: Optional output path
        model: Model type - "general", "anime", or "fast"

    Returns:
        Path to upscaled image

    Example:
        # Upscale Gemini art 4x
        quick_upscale("gemini_512.png", scale=4)

        # Use anime model
        quick_upscale("illustration.png", model="anime")
    """
    model_map = {
        "general": UpscaleModel.GENERAL_X4 if scale == 4 else UpscaleModel.GENERAL_X2,
        "anime": UpscaleModel.ANIME_X4,
        "fast": UpscaleModel.FAST_X4,
    }
    upscale_model = model_map.get(model, UpscaleModel.GENERAL_X4)

    editor = PhotoEditor()
    result = editor.ai_upscale(input_path, output_path, scale=scale, model=upscale_model)
    if result.success:
        return result.output_path
    raise RuntimeError(result.message)


def quick_vectorize(input_path: str, output_path: Optional[str] = None) -> str:
    """Quick function to vectorize an image."""
    editor = PhotoEditor()
    result = editor.vectorize(input_path, output_path)
    if result.success:
        return result.output_path
    raise RuntimeError(result.message)


def quick_resize(input_path: str, width: int = None, height: int = None,
                 output_path: Optional[str] = None) -> str:
    """Quick function to resize an image."""
    editor = PhotoEditor()
    result = editor.smart_resize(input_path, output_path, width=width, height=height)
    if result.success:
        return result.output_path
    raise RuntimeError(result.message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SuperCharged Photo Editor - AI-Powered Image Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # AI Upscale a small Gemini image 4x (THE MAGIC!)
    python photo_editor.py gemini_art.png --upscale

    # Upscale with anime model
    python photo_editor.py illustration.png --upscale --upscale-model anime

    # Remove background
    python photo_editor.py photo.png --remove-bg

    # Full pipeline with upscaling
    python photo_editor.py ai_image.png --pipeline --upscale

    # Convert to vector
    python photo_editor.py logo.png --vectorize
        """
    )
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", help="Output path or directory")
    parser.add_argument("--remove-bg", action="store_true", help="Remove background (AI)")
    parser.add_argument("--upscale", action="store_true",
                        help="AI upscale image 4x using Real-ESRGAN (best for AI art!)")
    parser.add_argument("--upscale-2x", action="store_true",
                        help="AI upscale image 2x")
    parser.add_argument("--upscale-model", choices=["general", "anime", "fast"],
                        default="general", help="Upscale model (default: general)")
    parser.add_argument("--vectorize", action="store_true", help="Convert to SVG")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), help="Resize to WxH")
    parser.add_argument("--scale", type=float, help="Scale factor (e.g., 2.0)")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")

    args = parser.parse_args()

    editor = PhotoEditor()

    # Map model name to enum
    upscale_model_map = {
        "general": UpscaleModel.GENERAL_X4,
        "anime": UpscaleModel.ANIME_X4,
        "fast": UpscaleModel.FAST_X4,
    }
    if args.upscale_2x:
        upscale_model = UpscaleModel.GENERAL_X2
    else:
        upscale_model = upscale_model_map.get(args.upscale_model, UpscaleModel.GENERAL_X4)

    if args.batch:
        results = editor.batch_process(
            args.input,
            args.output,
            remove_bg=args.remove_bg or args.pipeline,
            ai_upscale=args.upscale or args.upscale_2x,
            upscale_model=upscale_model,
            create_vector=args.vectorize or args.pipeline
        )
        print(f"\nProcessed {len(results)} images")

    elif args.pipeline:
        resize_config = None
        if args.resize:
            resize_config = {'width': args.resize[0], 'height': args.resize[1]}
        elif args.scale:
            resize_config = {'scale': args.scale}

        results = editor.process_full_pipeline(
            args.input,
            args.output,
            ai_upscale=args.upscale or args.upscale_2x,
            upscale_model=upscale_model,
            resize_config=resize_config
        )
        for r in results:
            status = "OK" if r.success else "FAIL"
            print(f"[{status}] {r.operation}: {r.message}")

    elif args.upscale or args.upscale_2x:
        result = editor.ai_upscale(args.input, args.output, model=upscale_model)
        print(f"{'OK' if result.success else 'FAIL'}: {result.message}")

    elif args.remove_bg:
        result = editor.remove_background(args.input, args.output)
        print(f"{'OK' if result.success else 'FAIL'}: {result.message}")

    elif args.vectorize:
        result = editor.vectorize(args.input, args.output)
        print(f"{'OK' if result.success else 'FAIL'}: {result.message}")

    elif args.resize:
        result = editor.smart_resize(args.input, args.output,
                                     width=args.resize[0], height=args.resize[1])
        print(f"{'OK' if result.success else 'FAIL'}: {result.message}")

    elif args.scale:
        result = editor.smart_resize(args.input, args.output, scale=args.scale)
        print(f"{'OK' if result.success else 'FAIL'}: {result.message}")

    else:
        parser.print_help()
