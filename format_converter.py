"""
Format Converter - Convert images to any format with one command

Drop an image and convert it to any preset format:
- "make this a banner"
- "make this a button"
- "make this a postcard"
- "make this a flyer"

Usage:
    python format_converter.py image.png banner
    python format_converter.py image.png postcard flyer icon
    python format_converter.py image.png --all-social
    python format_converter.py image.png --custom 800x600

Examples:
    # Single format
    python format_converter.py my_ai_art.png banner

    # Multiple formats at once
    python format_converter.py logo.png icon favicon app_icon

    # All social media sizes
    python format_converter.py promo.png --all-social

    # Custom size
    python format_converter.py image.png --custom 800x600

    # List all available presets
    python format_converter.py --list
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from PIL import Image

from presets import (
    Preset, FitMode, get_preset, list_presets, create_custom_preset,
    ALL_PRESETS, SOCIAL_PRESETS, PRINT_PRESETS, WEB_PRESETS
)


@dataclass
class ConversionResult:
    """Result of a format conversion."""
    success: bool
    input_path: str
    output_path: Optional[str]
    preset_name: str
    message: str
    original_size: Optional[Tuple[int, int]] = None
    output_size: Optional[Tuple[int, int]] = None


class FormatConverter:
    """
    Convert images to various preset formats.

    Example:
        converter = FormatConverter(output_base="./output")

        # Convert to a single format
        result = converter.convert("image.png", "banner")

        # Convert to multiple formats
        results = converter.convert_multiple("image.png", ["banner", "icon", "thumbnail"])

        # Convert to all social media formats
        results = converter.convert_to_all_social("image.png")
    """

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}

    def __init__(self, output_base: str = "output"):
        """
        Initialize the converter.

        Args:
            output_base: Base directory for all output files
        """
        self.output_base = Path(output_base)

    def convert(
        self,
        input_path: Union[str, Path],
        preset_name: str,
        output_path: Optional[Union[str, Path]] = None,
        upscale_if_needed: bool = True
    ) -> ConversionResult:
        """
        Convert an image to a specific preset format.

        Args:
            input_path: Path to input image
            preset_name: Name of the preset (e.g., "banner", "postcard")
            output_path: Optional custom output path
            upscale_if_needed: If True, upscale small images to fit

        Returns:
            ConversionResult with details
        """
        input_path = Path(input_path)

        # Get preset
        preset = get_preset(preset_name)
        if preset is None:
            return ConversionResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                preset_name=preset_name,
                message=f"Unknown preset: {preset_name}. Use --list to see available presets."
            )

        # Determine output path
        if output_path is None:
            output_dir = self.output_base / preset.output_folder
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}{preset.suffix}.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Open image
            img = Image.open(input_path)
            original_size = img.size

            # Convert to RGB/RGBA as needed
            if img.mode == 'P':
                img = img.convert('RGBA')
            elif img.mode == 'L':
                img = img.convert('RGB')

            # Apply the preset transformation
            result_img = self._apply_preset(img, preset)

            # Save
            if result_img.mode == 'RGBA' and output_path.suffix.lower() in ['.jpg', '.jpeg']:
                # Convert RGBA to RGB for JPEG
                background = Image.new('RGB', result_img.size, (255, 255, 255))
                background.paste(result_img, mask=result_img.split()[3])
                result_img = background

            # Set DPI for print formats
            save_kwargs = {'optimize': True}
            if preset.dpi > 72:
                save_kwargs['dpi'] = (preset.dpi, preset.dpi)

            result_img.save(output_path, **save_kwargs)

            return ConversionResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                preset_name=preset_name,
                message=f"Converted to {preset.name} ({preset.width}x{preset.height})",
                original_size=original_size,
                output_size=result_img.size
            )

        except Exception as e:
            return ConversionResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                preset_name=preset_name,
                message=f"Error: {str(e)}"
            )

    def _apply_preset(self, img: Image.Image, preset: Preset) -> Image.Image:
        """Apply a preset transformation to an image."""
        target_width, target_height = preset.width, preset.height
        orig_width, orig_height = img.size

        if preset.fit_mode == FitMode.CROP:
            return self._crop_to_fit(img, target_width, target_height)
        elif preset.fit_mode == FitMode.FIT:
            return self._fit_with_padding(img, target_width, target_height, preset.background_color)
        elif preset.fit_mode == FitMode.STRETCH:
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        elif preset.fit_mode == FitMode.COVER:
            return self._crop_to_fit(img, target_width, target_height)
        else:
            return self._crop_to_fit(img, target_width, target_height)

    def _crop_to_fit(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Crop image to fill target dimensions (center crop)."""
        orig_width, orig_height = img.size
        target_ratio = target_width / target_height
        orig_ratio = orig_width / orig_height

        if orig_ratio > target_ratio:
            # Image is wider - crop sides
            new_width = int(target_ratio * orig_height)
            offset = (orig_width - new_width) // 2
            crop_box = (offset, 0, offset + new_width, orig_height)
        else:
            # Image is taller - crop top/bottom
            new_height = int(orig_width / target_ratio)
            offset = (orig_height - new_height) // 2
            crop_box = (0, offset, orig_width, offset + new_height)

        cropped = img.crop(crop_box)
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def _fit_with_padding(
        self,
        img: Image.Image,
        target_width: int,
        target_height: int,
        bg_color: Tuple[int, int, int]
    ) -> Image.Image:
        """Fit image inside target dimensions with padding (letterbox)."""
        # Resize to fit
        img_copy = img.copy()
        img_copy.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        resized_width, resized_height = img_copy.size

        # Create background
        if img_copy.mode == 'RGBA':
            background = Image.new('RGBA', (target_width, target_height), (*bg_color, 255))
        else:
            background = Image.new('RGB', (target_width, target_height), bg_color)

        # Center and paste
        paste_x = (target_width - resized_width) // 2
        paste_y = (target_height - resized_height) // 2

        if img_copy.mode == 'RGBA':
            background.paste(img_copy, (paste_x, paste_y), img_copy)
        else:
            background.paste(img_copy, (paste_x, paste_y))

        return background

    def convert_multiple(
        self,
        input_path: Union[str, Path],
        preset_names: List[str]
    ) -> List[ConversionResult]:
        """Convert an image to multiple preset formats."""
        results = []
        for preset_name in preset_names:
            result = self.convert(input_path, preset_name)
            results.append(result)
            status = "OK" if result.success else "FAIL"
            print(f"[{status}] {preset_name}: {result.message}")
        return results

    def convert_to_all_social(self, input_path: Union[str, Path]) -> List[ConversionResult]:
        """Convert an image to all social media formats."""
        return self.convert_multiple(input_path, list(SOCIAL_PRESETS.keys()))

    def convert_to_all_print(self, input_path: Union[str, Path]) -> List[ConversionResult]:
        """Convert an image to all print formats."""
        return self.convert_multiple(input_path, list(PRINT_PRESETS.keys()))

    def convert_to_all_web(self, input_path: Union[str, Path]) -> List[ConversionResult]:
        """Convert an image to all web formats."""
        return self.convert_multiple(input_path, list(WEB_PRESETS.keys()))

    def convert_custom(
        self,
        input_path: Union[str, Path],
        width: int,
        height: int,
        fit_mode: FitMode = FitMode.CROP
    ) -> ConversionResult:
        """Convert an image to a custom size."""
        preset = create_custom_preset(
            name=f"{width}x{height}",
            width=width,
            height=height,
            fit_mode=fit_mode
        )

        input_path = Path(input_path)
        output_dir = self.output_base / "custom"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_{width}x{height}.png"

        try:
            img = Image.open(input_path)
            original_size = img.size

            if img.mode == 'P':
                img = img.convert('RGBA')

            result_img = self._apply_preset(img, preset)
            result_img.save(output_path, optimize=True)

            return ConversionResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                preset_name=f"custom_{width}x{height}",
                message=f"Converted to custom {width}x{height}",
                original_size=original_size,
                output_size=result_img.size
            )
        except Exception as e:
            return ConversionResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                preset_name=f"custom_{width}x{height}",
                message=f"Error: {str(e)}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert images to preset formats (banner, button, postcard, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert to a banner
    python format_converter.py image.png banner

    # Convert to multiple formats
    python format_converter.py image.png banner postcard icon

    # All social media sizes
    python format_converter.py image.png --all-social

    # All print sizes
    python format_converter.py image.png --all-print

    # Custom size
    python format_converter.py image.png --custom 800x600

    # List all presets
    python format_converter.py --list

Common shortcuts:
    banner     → Twitter banner (1500x500)
    button     → Medium button (200x60)
    icon       → Icon 256x256
    thumbnail  → Thumbnail 300x300
    postcard   → Postcard 6x4 (300 DPI)
    flyer      → Flyer letter size (300 DPI)
    poster     → Poster 11x17 (300 DPI)
    hero       → Hero image 1920x1080
        """
    )

    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("formats", nargs="*", help="Format presets to convert to")
    parser.add_argument("-o", "--output", help="Output directory (default: ./output)")
    parser.add_argument("--list", action="store_true", help="List all available presets")
    parser.add_argument("--all-social", action="store_true", help="Convert to all social media formats")
    parser.add_argument("--all-print", action="store_true", help="Convert to all print formats")
    parser.add_argument("--all-web", action="store_true", help="Convert to all web formats")
    parser.add_argument("--custom", metavar="WxH", help="Custom size (e.g., 800x600)")
    parser.add_argument("--fit", choices=["crop", "fit", "stretch"],
                        default="crop", help="Fit mode for custom size")

    args = parser.parse_args()

    # List presets
    if args.list:
        list_presets()
        return

    # Validate input
    if not args.input:
        parser.print_help()
        return

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    # Initialize converter
    output_dir = args.output or "output"
    converter = FormatConverter(output_base=output_dir)

    print(f"\n📸 Converting: {args.input}")
    print(f"📁 Output to: {output_dir}/")
    print("-" * 50)

    results = []

    # Custom size
    if args.custom:
        try:
            width, height = map(int, args.custom.lower().split("x"))
            fit_mode = {"crop": FitMode.CROP, "fit": FitMode.FIT, "stretch": FitMode.STRETCH}[args.fit]
            result = converter.convert_custom(args.input, width, height, fit_mode)
            results.append(result)
            status = "OK" if result.success else "FAIL"
            print(f"[{status}] custom {width}x{height}: {result.message}")
        except ValueError:
            print(f"Error: Invalid custom size format. Use WxH (e.g., 800x600)")
            sys.exit(1)

    # All social
    elif args.all_social:
        results = converter.convert_to_all_social(args.input)

    # All print
    elif args.all_print:
        results = converter.convert_to_all_print(args.input)

    # All web
    elif args.all_web:
        results = converter.convert_to_all_web(args.input)

    # Specific formats
    elif args.formats:
        results = converter.convert_multiple(args.input, args.formats)

    else:
        print("No format specified. Use --list to see available presets.")
        parser.print_help()
        return

    # Summary
    print("-" * 50)
    success_count = sum(1 for r in results if r.success)
    print(f"✅ Completed: {success_count}/{len(results)} formats converted")

    if success_count > 0:
        print(f"\n📂 Output files in: {output_dir}/")


if __name__ == "__main__":
    main()
