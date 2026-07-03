"""
Photo Editor - AI-powered image processing.

Features:
- Background removal (rembg / U2-Net)
- AI upscaling 2x-4x (Real-ESRGAN)
- Watermark removal via inpainting (LaMa, with OpenCV fallback)
- Raster to vector conversion (vtracer)
- Smart resizing with multiple algorithms
- Full pipeline and batch processing

Heavy AI libraries are imported lazily, so importing this module is cheap
and each feature only pays for what it uses.

Usage:
    from photo_editor import PhotoEditor

    editor = PhotoEditor()
    editor.remove_background("input.png", "output.png")
    editor.ai_upscale("small.png", "large.png", scale=4)
    editor.vectorize("input.png", "output.svg")
    editor.smart_resize("input.png", "output.png", width=1024)
"""

import functools
import io
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


# ============================================================================
# Lazy loaders for heavy AI libraries
# ============================================================================

_rembg = None
_vtracer = None
_cv2 = None
_lama = None
_upscalers: dict[tuple, object] = {}

# Real-ESRGAN model architectures: name -> (num_block, scale, weights URL)
_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_ESRGAN_MODELS = {
    "RealESRGAN_x4plus": (23, 4, f"{_ESRGAN_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth"),
    "RealESRGAN_x4plus_anime_6B": (6, 4, f"{_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    "RealESRGAN_x2plus": (23, 2, f"{_ESRGAN_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth"),
}


def _get_rembg():
    """Load rembg on first use."""
    global _rembg
    if _rembg is None:
        from rembg import new_session, remove
        _rembg = {"remove": remove, "new_session": new_session}
    return _rembg


def _get_vtracer():
    """Load vtracer on first use."""
    global _vtracer
    if _vtracer is None:
        import vtracer
        _vtracer = vtracer
    return _vtracer


def _get_cv2():
    """Load OpenCV on first use."""
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_lama():
    """Load the LaMa inpainting model, or fall back to OpenCV inpainting."""
    global _lama
    if _lama is None:
        try:
            from simple_lama_inpainting import SimpleLama
            _lama = SimpleLama()
            print("[LaMa] Inpainting model loaded")
        except ImportError:
            print("[LaMa] Not available, using OpenCV inpainting fallback")
            _lama = "opencv_fallback"
    return _lama


def _shim_torchvision():
    """Newer torchvision dropped functional_tensor; basicsr still imports it."""
    import sys

    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    import torchvision.transforms.functional as functional

    class _FunctionalTensorShim:
        rgb_to_grayscale = functional.rgb_to_grayscale

    sys.modules["torchvision.transforms.functional_tensor"] = _FunctionalTensorShim()


def _pick_device() -> tuple[str, bool]:
    """Best available torch device and whether FP16 is safe on it."""
    import torch

    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False  # MPS FP16 support is still unreliable
    return "cpu", False


def _get_upscaler(model_name: str, denoise_strength: float):
    """Build (and cache) a Real-ESRGAN upscaler for the given model."""
    cache_key = (model_name, denoise_strength)
    if cache_key in _upscalers:
        return _upscalers[cache_key]

    _shim_torchvision()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as e:
        raise RuntimeError(
            f"Real-ESRGAN not installed. Run: pip install realesrgan basicsr ({e})"
        ) from e

    device, half_precision = _pick_device()
    print(f"[Real-ESRGAN] Using device: {device}")

    num_block, netscale, model_url = _ESRGAN_MODELS.get(
        model_name, _ESRGAN_MODELS["RealESRGAN_x4plus"]
    )
    upscaler = RealESRGANer(
        scale=netscale,
        model_path=model_url,
        dni_weight=denoise_strength,
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=num_block, num_grow_ch=32, scale=netscale),
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
        device=device,
    )
    _upscalers[cache_key] = upscaler
    return upscaler


# ============================================================================
# Options and results
# ============================================================================

class ResizeMode(Enum):
    """Resize algorithm options."""

    LANCZOS = "lanczos"    # Best for downscaling, sharp results
    BICUBIC = "bicubic"    # Good balance of quality and speed
    BILINEAR = "bilinear"  # Fast, good for real-time
    NEAREST = "nearest"    # Pixel-perfect, good for pixel art
    SUPERRES = "superres"  # OpenCV cubic interpolation


class UpscaleModel(Enum):
    """AI upscaling model options (Real-ESRGAN)."""

    GENERAL_X4 = "RealESRGAN_x4plus"         # Best quality, 4x
    GENERAL_X2 = "RealESRGAN_x2plus"         # 2x
    ANIME_X4 = "RealESRGAN_x4plus_anime_6B"  # Anime / illustrations, 4x
    FAST_X4 = "realesr-general-x4v3"         # Faster, good quality, 4x

    @classmethod
    def from_name(cls, name: str, scale: int = 4) -> "UpscaleModel":
        """Resolve a friendly name ("general", "anime", "fast") and scale."""
        special = {"anime": cls.ANIME_X4, "fast": cls.FAST_X4}
        if name in special:
            return special[name]
        return cls.GENERAL_X2 if scale == 2 else cls.GENERAL_X4

    @property
    def scale(self) -> int:
        return 2 if self is UpscaleModel.GENERAL_X2 else 4


class VectorMode(Enum):
    """Vectorization style presets."""

    PHOTO = "photo"
    ILLUSTRATION = "illustration"
    LOGO = "logo"
    PIXEL_ART = "pixel_art"


# vtracer parameter overrides per vectorization style
_VECTOR_STYLE_OVERRIDES = {
    VectorMode.LOGO: {"filter_speckle": 8, "color_precision": 4, "corner_threshold": 90},
    VectorMode.PIXEL_ART: {"filter_speckle": 0, "color_precision": 8,
                           "corner_threshold": 0, "length_threshold": 0},
    VectorMode.PHOTO: {"filter_speckle": 2, "color_precision": 8, "layer_difference": 8},
}

_RESAMPLING = {
    ResizeMode.LANCZOS: Image.Resampling.LANCZOS,
    ResizeMode.BICUBIC: Image.Resampling.BICUBIC,
    ResizeMode.BILINEAR: Image.Resampling.BILINEAR,
    ResizeMode.NEAREST: Image.Resampling.NEAREST,
}


@dataclass
class ProcessingResult:
    """Result of an image processing operation."""

    success: bool
    input_path: str
    output_path: str | None
    operation: str
    message: str
    original_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    file_size_before: int | None = None
    file_size_after: int | None = None


def _operation(name: str):
    """Turn any exception in a PhotoEditor method into a failure result."""
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, input_path: PathLike, *args, **kwargs):
            try:
                return method(self, Path(input_path), *args, **kwargs)
            except Exception as e:
                return ProcessingResult(
                    success=False, input_path=str(input_path), output_path=None,
                    operation=name, message=f"Error: {e}",
                )
        return wrapper
    return decorator


# ============================================================================
# Image helpers
# ============================================================================

def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a white background (for JPEG output)."""
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.getchannel("A"))
    return background


def _save_image(img: Image.Image, path: Path,
                output_format: str | None = None, quality: int = 95) -> Image.Image:
    """Save an image, inferring format from the path unless given explicitly.

    Handles JPG->JPEG aliasing, lossy quality, and alpha flattening for JPEG.
    Returns the image actually saved (flattened if needed).
    """
    fmt = (output_format or path.suffix.lstrip(".") or "png").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    save_kwargs = {"optimize": True}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        if fmt == "JPEG" and img.mode == "RGBA":
            img = _flatten_to_rgb(img)

    img.save(path, format=fmt, **save_kwargs)
    return img


def _watermark_box(position: str, img_width: int, img_height: int,
                   box_width: int, box_height: int) -> tuple[int, int, int, int]:
    """Pixel box (x1, y1, x2, y2) for a watermark at a named corner/edge.

    Positions: top/bottom x left/center/right, e.g. "bottom-right" (default
    for anything unrecognized).
    """
    vertical, _, horizontal = position.partition("-")
    y1 = 0 if vertical == "top" else img_height - box_height
    if horizontal == "left":
        x1 = 0
    elif horizontal == "center":
        x1 = (img_width - box_width) // 2
    else:
        x1 = img_width - box_width
    return x1, y1, x1 + box_width, y1 + box_height


def _target_size(original: tuple[int, int],
                 width: int | None, height: int | None, scale: float | None,
                 maintain_aspect: bool, max_size: int | None) -> tuple[int, int]:
    """Resolve resize parameters into concrete output dimensions."""
    orig_width, orig_height = original

    if scale is not None:
        return int(orig_width * scale), int(orig_height * scale)
    if width is not None and height is not None:
        if not maintain_aspect:
            return width, height
        ratio = min(width / orig_width, height / orig_height)
        return int(orig_width * ratio), int(orig_height * ratio)
    if width is not None:
        return width, int(orig_height * width / orig_width)
    if height is not None:
        return int(orig_width * height / orig_height), height
    if max_size is not None and max(original) > max_size:
        ratio = max_size / max(original)
        return int(orig_width * ratio), int(orig_height * ratio)
    return orig_width, orig_height


# ============================================================================
# PhotoEditor
# ============================================================================

class PhotoEditor:
    """AI-powered photo editing toolkit.

    Every operation returns a :class:`ProcessingResult` instead of raising,
    so batch flows can keep going after individual failures.

    Example:
        editor = PhotoEditor()
        editor.remove_background("photo.png", "photo_nobg.png")
        editor.ai_upscale("small_art.png", "large_art.png", scale=4)
        editor.vectorize("logo.png", "logo.svg")
        editor.process_full_pipeline("ai_image.png", "output/")
    """

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

    def __init__(self, default_output_dir: PathLike = "processed",
                 rembg_model: str = "u2net"):
        """
        Args:
            default_output_dir: Default directory for pipeline output.
            rembg_model: Background removal model - "u2net" (best quality),
                "u2netp" (faster), "u2net_human_seg" (people),
                "isnet-general-use", or "isnet-anime".
        """
        self.default_output_dir = Path(default_output_dir)
        self.rembg_model = rembg_model
        self._rembg_session = None

    # -- internal helpers ---------------------------------------------------

    def _get_rembg_session(self):
        """Reuse one rembg session across calls for faster batches."""
        if self._rembg_session is None:
            self._rembg_session = _get_rembg()["new_session"](self.rembg_model)
        return self._rembg_session

    @staticmethod
    def _resolve_output(input_path: Path, output_path: PathLike | None,
                        default_name: str) -> Path:
        """Pick the output path (next to the input by default) and make its dir."""
        path = Path(output_path) if output_path else input_path.parent / default_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _success(operation: str, input_path: Path, output_path: Path, message: str,
                 original_size: tuple[int, int],
                 output_size: tuple[int, int]) -> ProcessingResult:
        return ProcessingResult(
            success=True,
            input_path=str(input_path),
            output_path=str(output_path),
            operation=operation,
            message=message,
            original_size=original_size,
            output_size=output_size,
            file_size_before=os.path.getsize(input_path),
            file_size_after=os.path.getsize(output_path),
        )

    # -- operations ----------------------------------------------------------

    @_operation("remove_background")
    def remove_background(self, input_path: Path,
                          output_path: PathLike | None = None,
                          alpha_matting: bool = False,
                          alpha_matting_foreground_threshold: int = 240,
                          alpha_matting_background_threshold: int = 10,
                          only_mask: bool = False) -> ProcessingResult:
        """Remove the background using AI (rembg / U2-Net).

        Args:
            input_path: Path to input image.
            output_path: Output path (default: ``<input>_nobg.png``).
            alpha_matting: Enable alpha matting for better edges.
            alpha_matting_foreground_threshold: Foreground threshold (0-255).
            alpha_matting_background_threshold: Background threshold (0-255).
            only_mask: Output only the mask instead of the image.
        """
        output_path = self._resolve_output(input_path, output_path,
                                           f"{input_path.stem}_nobg.png")

        original_size = Image.open(input_path).size
        output_data = _get_rembg()["remove"](
            input_path.read_bytes(),
            session=self._get_rembg_session(),
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            only_mask=only_mask,
        )
        output_img = Image.open(io.BytesIO(output_data))
        output_img.save(output_path, format="PNG", optimize=True)

        return self._success("remove_background", input_path, output_path,
                             "Background removed", original_size, output_img.size)

    @_operation("ai_upscale")
    def ai_upscale(self, input_path: Path,
                   output_path: PathLike | None = None,
                   scale: int = 4,
                   model: UpscaleModel = UpscaleModel.GENERAL_X4,
                   denoise_strength: float = 0.5,
                   output_format: str | None = None,
                   quality: int = 95) -> ProcessingResult:
        """AI-upscale an image with Real-ESRGAN - adds detail instead of blur.

        Ideal for enlarging small AI-generated images (Gemini, DALL-E,
        Midjourney, ...). The effective scale comes from the model: 2x for
        GENERAL_X2, otherwise 4x.

        Args:
            input_path: Path to input image.
            output_path: Output path (default: ``<input>_upscaled_<N>x.png``).
            scale: Requested factor, used for default naming (2 or 4).
            model: GENERAL_X4, GENERAL_X2, ANIME_X4, or FAST_X4.
            denoise_strength: 0.0-1.0, higher removes more noise.
            output_format: png, jpg, or webp (default: from output path).
            quality: JPEG/WebP quality (1-100).
        """
        scale = model.scale
        output_path = self._resolve_output(input_path, output_path,
                                           f"{input_path.stem}_upscaled_{scale}x.png")

        img = Image.open(input_path)
        original_size = img.size
        has_alpha = img.mode == "RGBA"

        # Real-ESRGAN works on OpenCV-style BGR arrays
        cv2 = _get_cv2()
        array = np.array(img if has_alpha else img.convert("RGB"))
        bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA if has_alpha else cv2.COLOR_RGB2BGR)

        upscaler = _get_upscaler(model.value, denoise_strength)
        output_bgr, _ = upscaler.enhance(bgr, outscale=scale)

        rgb = cv2.cvtColor(output_bgr,
                           cv2.COLOR_BGRA2RGBA if has_alpha else cv2.COLOR_BGR2RGB)
        output_img = _save_image(Image.fromarray(rgb), output_path,
                                 output_format, quality)

        message = (f"Upscaled {scale}x: {original_size[0]}x{original_size[1]} -> "
                   f"{output_img.size[0]}x{output_img.size[1]} ({model.value})")
        return self._success("ai_upscale", input_path, output_path, message,
                             original_size, output_img.size)

    @_operation("vectorize")
    def vectorize(self, input_path: Path,
                  output_path: PathLike | None = None,
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
        """Convert a raster image to vector SVG using vtracer.

        The ``mode`` preset tunes the tracing parameters for the content type;
        explicit keyword arguments fill in everything the preset doesn't set.

        Args:
            input_path: Path to input image.
            output_path: Output path (default: ``<input>.svg``).
            mode: PHOTO, ILLUSTRATION, LOGO, or PIXEL_ART.
            colormode: "color" or "binary".
            hierarchical: "stacked" or "cutout".
            filter_speckle: Speckle filter size.
            color_precision: Color precision (1-8).
            layer_difference: Layer difference threshold.
            corner_threshold: Corner detection threshold.
            length_threshold: Minimum path length.
            max_iterations: Max curve fitting iterations.
            splice_threshold: Path splicing threshold.
            path_precision: SVG path precision.
        """
        output_path = self._resolve_output(input_path, output_path,
                                           f"{input_path.stem}.svg")

        img = Image.open(input_path)
        original_size = img.size
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        params = {
            "colormode": colormode,
            "hierarchical": hierarchical,
            "mode": "polygon",
            "filter_speckle": filter_speckle,
            "color_precision": color_precision,
            "layer_difference": layer_difference,
            "corner_threshold": corner_threshold,
            "length_threshold": length_threshold,
            "max_iterations": max_iterations,
            "splice_threshold": splice_threshold,
            "path_precision": path_precision,
            **_VECTOR_STYLE_OVERRIDES.get(mode, {}),
        }
        svg = _get_vtracer().convert_raw_image_to_svg(
            buffer.getvalue(), img_format="png", **params
        )
        output_path.write_text(svg, encoding="utf-8")

        return self._success("vectorize", input_path, output_path,
                             f"Vectorized to SVG ({mode.value} mode)",
                             original_size, original_size)

    @_operation("remove_watermark")
    def remove_watermark(self, input_path: Path,
                         output_path: PathLike | None = None,
                         watermark_position: str = "bottom-right",
                         watermark_height: int = 50,
                         watermark_width: int = 200,
                         padding: int = 10,
                         use_lama: bool = True) -> ProcessingResult:
        """Remove a corner watermark (like Gemini's) via AI inpainting.

        Uses LaMa (Large Mask Inpainting) when installed, otherwise falls
        back to OpenCV's TELEA inpainting.

        Args:
            input_path: Path to input image.
            output_path: Output path (default: ``<input>_nowm.png``).
            watermark_position: "bottom-right" (default), "bottom-left",
                "top-right", "top-left", or "bottom-center".
            watermark_height: Height of the watermark region in pixels.
            watermark_width: Width of the watermark region in pixels.
            padding: Extra padding around the region.
            use_lama: Prefer LaMa over the OpenCV fallback.
        """
        output_path = self._resolve_output(input_path, output_path,
                                           f"{input_path.stem}_nowm.png")

        img = Image.open(input_path).convert("RGB")
        width, height = img.size

        # Mask the watermark region (white = inpaint), capped to a sane
        # fraction of the image so oversized boxes can't eat the picture.
        box_h = min(watermark_height + padding * 2, height // 4)
        box_w = min(watermark_width + padding * 2, width // 3)
        x1, y1, x2, y2 = _watermark_box(watermark_position, width, height, box_w, box_h)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        lama = _get_lama() if use_lama else "opencv_fallback"
        if lama != "opencv_fallback":
            result_img = lama(img, Image.fromarray(mask))
        else:
            cv2 = _get_cv2()
            result_array = cv2.inpaint(np.array(img), mask,
                                       inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            result_img = Image.fromarray(result_array)

        result_img.save(output_path, format="PNG", optimize=True)

        return self._success("remove_watermark", input_path, output_path,
                             f"Watermark removed from {watermark_position}",
                             img.size, result_img.size)

    @_operation("smart_resize")
    def smart_resize(self, input_path: Path,
                     output_path: PathLike | None = None,
                     width: int | None = None,
                     height: int | None = None,
                     scale: float | None = None,
                     mode: ResizeMode = ResizeMode.LANCZOS,
                     maintain_aspect: bool = True,
                     max_size: int | None = None,
                     output_format: str | None = None,
                     quality: int = 95) -> ProcessingResult:
        """Resize an image using the given algorithm.

        Args:
            input_path: Path to input image.
            output_path: Output path (default derived from dimensions).
            width: Target width.
            height: Target height.
            scale: Scale factor, e.g. 2.0 (alternative to width/height).
            mode: LANCZOS, BICUBIC, BILINEAR, NEAREST, or SUPERRES.
            maintain_aspect: Fit within width x height, keeping proportions.
            max_size: Cap the longest side (scales down only).
            output_format: png, jpg, or webp (default: from output path).
            quality: JPEG/WebP quality (1-100).
        """
        default_suffix = f"_{width}x{height}" if width and height else "_resized"
        output_path = self._resolve_output(
            input_path, output_path,
            f"{input_path.stem}{default_suffix}{input_path.suffix}",
        )

        img = Image.open(input_path)
        original_size = img.size
        new_size = _target_size(original_size, width, height, scale,
                                maintain_aspect, max_size)

        if mode == ResizeMode.SUPERRES:
            resized = self._superres_resize(img, *new_size)
        else:
            resample = _RESAMPLING.get(mode, Image.Resampling.LANCZOS)
            resized = img.resize(new_size, resample)

        _save_image(resized, output_path, output_format, quality)

        message = (f"Resized from {original_size[0]}x{original_size[1]} "
                   f"to {new_size[0]}x{new_size[1]} ({mode.value})")
        return self._success("smart_resize", input_path, output_path, message,
                             original_size, new_size)

    @staticmethod
    def _superres_resize(img: Image.Image, width: int, height: int) -> Image.Image:
        """Resize via OpenCV cubic interpolation; LANCZOS if OpenCV is missing."""
        try:
            cv2 = _get_cv2()
        except ImportError:
            return img.resize((width, height), Image.Resampling.LANCZOS)

        array = np.array(img)
        color = array.ndim == 3
        has_alpha = color and array.shape[2] == 4

        if has_alpha:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
        elif color:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

        resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_CUBIC)

        if has_alpha:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)
        elif color:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(resized)

    # -- pipelines ------------------------------------------------------------

    def process_full_pipeline(self, input_path: PathLike,
                              output_dir: PathLike | None = None,
                              remove_bg: bool = True,
                              ai_upscale: bool = False,
                              upscale_model: UpscaleModel = UpscaleModel.GENERAL_X4,
                              create_vector: bool = True,
                              resize_config: dict | None = None,
                              vector_mode: VectorMode = VectorMode.ILLUSTRATION,
                              ) -> list[ProcessingResult]:
        """Run the full pipeline: remove background -> upscale -> vectorize -> resize.

        Each enabled step feeds its output into the next; failed steps are
        recorded and skipped over.

        Args:
            input_path: Path to input image.
            output_dir: Output directory (default: ``<default_output_dir>/<stem>``).
            remove_bg: Remove the background.
            ai_upscale: AI-upscale with Real-ESRGAN.
            upscale_model: Which upscale model to use.
            create_vector: Also produce an SVG.
            resize_config: Resize parameters (width, height, scale, mode) or
                ``{"sizes": [...]}`` for multiple outputs.
            vector_mode: Vectorization style preset.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else self.default_output_dir / input_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[ProcessingResult] = []
        current = input_path

        def run(result: ProcessingResult, chain: bool = False) -> None:
            nonlocal current
            results.append(result)
            if chain and result.success:
                current = Path(result.output_path)

        if remove_bg:
            run(self.remove_background(current, output_dir / f"{input_path.stem}_nobg.png"),
                chain=True)

        if ai_upscale:
            scale = upscale_model.scale
            run(self.ai_upscale(current,
                                output_dir / f"{input_path.stem}_upscaled_{scale}x.png",
                                model=upscale_model),
                chain=True)

        if create_vector:
            run(self.vectorize(current, output_dir / f"{input_path.stem}.svg",
                               mode=vector_mode))

        if resize_config:
            sizes = resize_config.get("sizes", [resize_config])
            if not isinstance(sizes, list):
                sizes = [sizes]
            for i, config in enumerate(sizes):
                suffix = config.get("suffix", f"_{i}") if len(sizes) > 1 else "_resized"
                run(self.smart_resize(
                    current, output_dir / f"{input_path.stem}{suffix}.png",
                    width=config.get("width"), height=config.get("height"),
                    scale=config.get("scale"),
                    mode=config.get("mode", ResizeMode.LANCZOS),
                ))

        return results

    def batch_process(self, input_dir: PathLike,
                      output_dir: PathLike | None = None,
                      **pipeline_kwargs) -> list[list[ProcessingResult]]:
        """Run the full pipeline on every supported image in a directory."""
        all_results = []
        for file_path in Path(input_dir).iterdir():
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                continue

            img_output_dir = Path(output_dir) / file_path.stem if output_dir else None
            results = self.process_full_pipeline(file_path, output_dir=img_output_dir,
                                                 **pipeline_kwargs)
            all_results.append(results)

            for r in results:
                status = "OK" if r.success else "FAIL"
                print(f"[{status}] {r.operation}: {r.input_path} -> {r.output_path or r.message}")

        return all_results


# ============================================================================
# Quick one-shot helpers
# ============================================================================

def _unwrap(result: ProcessingResult) -> str:
    """Return the output path of a successful result, or raise."""
    if result.success:
        return result.output_path
    raise RuntimeError(result.message)


def quick_remove_bg(input_path: str, output_path: str | None = None) -> str:
    """Remove the background from an image; returns the output path."""
    return _unwrap(PhotoEditor().remove_background(input_path, output_path))


def quick_upscale(input_path: str, scale: int = 4, output_path: str | None = None,
                  model: str = "general") -> str:
    """AI-upscale an image (model: "general", "anime", or "fast")."""
    upscale_model = UpscaleModel.from_name(model, scale)
    return _unwrap(PhotoEditor().ai_upscale(input_path, output_path,
                                            scale=scale, model=upscale_model))


def quick_vectorize(input_path: str, output_path: str | None = None) -> str:
    """Convert an image to SVG; returns the output path."""
    return _unwrap(PhotoEditor().vectorize(input_path, output_path))


def quick_resize(input_path: str, width: int | None = None, height: int | None = None,
                 output_path: str | None = None) -> str:
    """Resize an image; returns the output path."""
    return _unwrap(PhotoEditor().smart_resize(input_path, output_path,
                                              width=width, height=height))


def quick_remove_watermark(input_path: str, output_path: str | None = None,
                           position: str = "bottom-right") -> str:
    """Remove a corner watermark; returns the output path."""
    return _unwrap(PhotoEditor().remove_watermark(input_path, output_path,
                                                  watermark_position=position))


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Photo Editor - AI-powered image processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python photo_editor.py gemini_art.png --upscale
    python photo_editor.py illustration.png --upscale --upscale-model anime
    python photo_editor.py photo.png --remove-bg
    python photo_editor.py ai_image.png --pipeline --upscale
    python photo_editor.py logo.png --vectorize
        """,
    )
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", help="Output path or directory")
    parser.add_argument("--remove-bg", action="store_true", help="Remove background (AI)")
    parser.add_argument("--upscale", action="store_true",
                        help="AI upscale 4x using Real-ESRGAN")
    parser.add_argument("--upscale-2x", action="store_true", help="AI upscale 2x")
    parser.add_argument("--upscale-model", choices=["general", "anime", "fast"],
                        default="general", help="Upscale model (default: general)")
    parser.add_argument("--vectorize", action="store_true", help="Convert to SVG")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        help="Resize to WxH")
    parser.add_argument("--scale", type=float, help="Scale factor (e.g., 2.0)")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    args = parser.parse_args()

    editor = PhotoEditor()
    upscale_model = UpscaleModel.from_name(args.upscale_model,
                                           scale=2 if args.upscale_2x else 4)

    def report(result: ProcessingResult) -> None:
        print(f"{'OK' if result.success else 'FAIL'}: {result.message}")

    if args.batch:
        results = editor.batch_process(
            args.input, args.output,
            remove_bg=args.remove_bg or args.pipeline,
            ai_upscale=args.upscale or args.upscale_2x,
            upscale_model=upscale_model,
            create_vector=args.vectorize or args.pipeline,
        )
        print(f"\nProcessed {len(results)} images")
    elif args.pipeline:
        resize_config = None
        if args.resize:
            resize_config = {"width": args.resize[0], "height": args.resize[1]}
        elif args.scale:
            resize_config = {"scale": args.scale}

        for result in editor.process_full_pipeline(
                args.input, args.output,
                ai_upscale=args.upscale or args.upscale_2x,
                upscale_model=upscale_model,
                resize_config=resize_config):
            print(f"[{'OK' if result.success else 'FAIL'}] {result.operation}: {result.message}")
    elif args.upscale or args.upscale_2x:
        report(editor.ai_upscale(args.input, args.output, model=upscale_model))
    elif args.remove_bg:
        report(editor.remove_background(args.input, args.output))
    elif args.vectorize:
        report(editor.vectorize(args.input, args.output))
    elif args.resize:
        report(editor.smart_resize(args.input, args.output,
                                   width=args.resize[0], height=args.resize[1]))
    elif args.scale:
        report(editor.smart_resize(args.input, args.output, scale=args.scale))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
