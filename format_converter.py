"""
Format Converter - convert images to any preset format with one command.

Usage:
    python format_converter.py image.png banner
    python format_converter.py image.png postcard flyer icon
    python format_converter.py image.png --all-social
    python format_converter.py image.png --custom 800x600
    python format_converter.py --list
"""

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union

from PIL import Image

from presets import (
    ALL_PRESETS, PRINT_PRESETS, SOCIAL_PRESETS, WEB_PRESETS,
    FitMode, Preset, create_custom_preset, get_preset, list_presets,
)

PathLike = Union[str, Path]


@dataclass
class ConversionResult:
    """Result of a format conversion."""

    success: bool
    input_path: str
    output_path: str | None
    preset_name: str
    message: str
    original_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None


class FormatConverter:
    """Convert images to preset formats (banner, postcard, icon, ...).

    Example:
        converter = FormatConverter(output_base="./output")
        converter.convert("image.png", "banner")
        converter.convert_multiple("image.png", ["banner", "icon", "thumbnail"])
        converter.convert_to_all_social("image.png")
    """

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

    def __init__(self, output_base: PathLike = "output"):
        """
        Args:
            output_base: Base directory for all output files.
        """
        self.output_base = Path(output_base)

    def convert(self, input_path: PathLike, preset_name: str,
                output_path: PathLike | None = None,
                fit_mode_override: FitMode | None = None,
                background_color: tuple[int, int, int] | None = None,
                ) -> ConversionResult:
        """Convert an image to a named preset format.

        Args:
            input_path: Path to input image.
            preset_name: Preset name or alias (e.g. "banner", "postcard").
            output_path: Custom output path (default: under ``output_base``).
            fit_mode_override: Override the preset's fit mode
                (e.g. ``FitMode.FIT`` for letterbox instead of crop).
            background_color: Letterbox background color override.
        """
        input_path = Path(input_path)

        preset = get_preset(preset_name)
        if preset is None:
            return ConversionResult(
                success=False, input_path=str(input_path), output_path=None,
                preset_name=preset_name,
                message=f"Unknown preset: {preset_name}. Use --list to see available presets.",
            )

        suffix = preset.suffix
        if fit_mode_override is not None:
            preset = replace(preset, fit_mode=fit_mode_override)
            if fit_mode_override == FitMode.FIT:
                suffix += "_fit"
        if background_color is not None:
            preset = replace(preset, background_color=background_color)

        if output_path is None:
            output_path = (self.output_base / preset.output_folder
                           / f"{input_path.stem}{suffix}.png")

        return self._convert(input_path, preset, Path(output_path), preset_name)

    def convert_custom(self, input_path: PathLike, width: int, height: int,
                       fit_mode: FitMode = FitMode.CROP) -> ConversionResult:
        """Convert an image to an arbitrary size."""
        input_path = Path(input_path)
        preset = create_custom_preset(f"{width}x{height}", width, height, fit_mode)
        output_path = self.output_base / "custom" / f"{input_path.stem}_{width}x{height}.png"
        return self._convert(input_path, preset, output_path, f"custom_{width}x{height}")

    def _convert(self, input_path: Path, preset: Preset,
                 output_path: Path, preset_name: str) -> ConversionResult:
        """Apply a preset and save; shared by named and custom conversions."""
        try:
            img = Image.open(input_path)
            original_size = img.size

            # Normalize palette/grayscale modes for compositing
            if img.mode == "P":
                img = img.convert("RGBA")
            elif img.mode == "L":
                img = img.convert("RGB")

            result_img = self._apply_preset(img, preset)

            if result_img.mode == "RGBA" and output_path.suffix.lower() in (".jpg", ".jpeg"):
                background = Image.new("RGB", result_img.size, (255, 255, 255))
                background.paste(result_img, mask=result_img.getchannel("A"))
                result_img = background

            save_kwargs = {"optimize": True}
            if preset.dpi > 72:
                save_kwargs["dpi"] = (preset.dpi, preset.dpi)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(output_path, **save_kwargs)

            return ConversionResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                preset_name=preset_name,
                message=f"Converted to {preset.name} ({preset.width}x{preset.height})",
                original_size=original_size,
                output_size=result_img.size,
            )
        except Exception as e:
            return ConversionResult(
                success=False, input_path=str(input_path), output_path=None,
                preset_name=preset_name, message=f"Error: {e}",
            )

    def _apply_preset(self, img: Image.Image, preset: Preset) -> Image.Image:
        """Resize/crop/pad an image to the preset's dimensions."""
        if preset.fit_mode == FitMode.FIT:
            return self._fit_with_padding(img, preset.width, preset.height,
                                          preset.background_color)
        if preset.fit_mode == FitMode.STRETCH:
            return img.resize(preset.size, Image.Resampling.LANCZOS)
        return self._crop_to_fit(img, preset.width, preset.height)  # CROP / COVER

    @staticmethod
    def _crop_to_fit(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Center-crop to the target aspect ratio, then resize to fill it."""
        orig_width, orig_height = img.size
        target_ratio = target_width / target_height

        if orig_width / orig_height > target_ratio:
            # Image is wider - crop the sides
            new_width = int(target_ratio * orig_height)
            offset = (orig_width - new_width) // 2
            crop_box = (offset, 0, offset + new_width, orig_height)
        else:
            # Image is taller - crop top and bottom
            new_height = int(orig_width / target_ratio)
            offset = (orig_height - new_height) // 2
            crop_box = (0, offset, orig_width, offset + new_height)

        return img.crop(crop_box).resize((target_width, target_height),
                                         Image.Resampling.LANCZOS)

    @staticmethod
    def _fit_with_padding(img: Image.Image, target_width: int, target_height: int,
                          bg_color: tuple[int, int, int]) -> Image.Image:
        """Fit the image inside the target box, padding the rest (letterbox)."""
        img = img.copy()
        img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

        if img.mode == "RGBA":
            background = Image.new("RGBA", (target_width, target_height), (*bg_color, 255))
        else:
            background = Image.new("RGB", (target_width, target_height), bg_color)

        paste_at = ((target_width - img.width) // 2, (target_height - img.height) // 2)
        background.paste(img, paste_at, img if img.mode == "RGBA" else None)
        return background

    def convert_multiple(self, input_path: PathLike,
                         preset_names: list[str]) -> list[ConversionResult]:
        """Convert an image to several preset formats, reporting each."""
        results = []
        for preset_name in preset_names:
            result = self.convert(input_path, preset_name)
            results.append(result)
            print(f"[{'OK' if result.success else 'FAIL'}] {preset_name}: {result.message}")
        return results

    def convert_to_all_social(self, input_path: PathLike) -> list[ConversionResult]:
        """Convert an image to every social media format."""
        return self.convert_multiple(input_path, list(SOCIAL_PRESETS))

    def convert_to_all_print(self, input_path: PathLike) -> list[ConversionResult]:
        """Convert an image to every print format."""
        return self.convert_multiple(input_path, list(PRINT_PRESETS))

    def convert_to_all_web(self, input_path: PathLike) -> list[ConversionResult]:
        """Convert an image to every web format."""
        return self.convert_multiple(input_path, list(WEB_PRESETS))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert images to preset formats (banner, button, postcard, ...)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python format_converter.py image.png banner
    python format_converter.py image.png banner postcard icon
    python format_converter.py image.png --all-social
    python format_converter.py image.png --custom 800x600
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
        """,
    )
    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("formats", nargs="*", help="Format presets to convert to")
    parser.add_argument("-o", "--output", help="Output directory (default: ./output)")
    parser.add_argument("--list", action="store_true", help="List all available presets")
    parser.add_argument("--all-social", action="store_true", help="All social media formats")
    parser.add_argument("--all-print", action="store_true", help="All print formats")
    parser.add_argument("--all-web", action="store_true", help="All web formats")
    parser.add_argument("--custom", metavar="WxH", help="Custom size (e.g., 800x600)")
    parser.add_argument("--fit", choices=["crop", "fit", "stretch"], default="crop",
                        help="Fit mode for custom size")
    args = parser.parse_args()

    if args.list:
        list_presets()
        return

    if not args.input:
        parser.print_help()
        return

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    output_dir = args.output or "output"
    converter = FormatConverter(output_base=output_dir)

    print(f"\n📸 Converting: {args.input}")
    print(f"📁 Output to: {output_dir}/")
    print("-" * 50)

    if args.custom:
        try:
            width, height = map(int, args.custom.lower().split("x"))
        except ValueError:
            print("Error: Invalid custom size format. Use WxH (e.g., 800x600)")
            sys.exit(1)
        result = converter.convert_custom(args.input, width, height, FitMode(args.fit))
        results = [result]
        print(f"[{'OK' if result.success else 'FAIL'}] custom {width}x{height}: {result.message}")
    elif args.all_social:
        results = converter.convert_to_all_social(args.input)
    elif args.all_print:
        results = converter.convert_to_all_print(args.input)
    elif args.all_web:
        results = converter.convert_to_all_web(args.input)
    elif args.formats:
        results = converter.convert_multiple(args.input, args.formats)
    else:
        print("No format specified. Use --list to see available presets.")
        parser.print_help()
        return

    print("-" * 50)
    success_count = sum(r.success for r in results)
    print(f"✅ Completed: {success_count}/{len(results)} formats converted")
    if success_count:
        print(f"\n📂 Output files in: {output_dir}/")


if __name__ == "__main__":
    main()
