#!/usr/bin/env python3
"""
Smart CLI - natural language photo editing.

Tell it what you want in plain English:

    python smart_cli.py "create a button from logo.png"
    python smart_cli.py "make banner from my_art.png"
    python smart_cli.py "upscale this image 4x" image.png
    python smart_cli.py "remove background from photo.jpg"
    python smart_cli.py "convert portrait.png to postcard"

Interactive mode:
    python smart_cli.py
    > create a banner from sunset.png
    > upscale dragon.png
    > exit
"""

import os
import re
import readline  # noqa: F401 - enables input history in interactive mode
from pathlib import Path
from typing import Optional

from format_converter import FormatConverter
from photo_editor import PhotoEditor, UpscaleModel
from presets import ALL_PRESETS, list_presets

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif")


class CommandParser:
    """Parse natural language commands and run the matching operation."""

    PATTERNS = {
        "upscale": [
            r"(?:upscale|upsize|enlarge|make.+bigger|make.+larger|scale.+up|increase.+size)",
            r"(?:(\d)x|(\d+)\s*times)",
        ],
        "remove_bg": [
            r"(?:remove.+background|remove.+bg|no.+background|transparent|cut.?out|extract)",
        ],
        "vectorize": [
            r"(?:vector|svg|vectorize|convert.+to.+svg|make.+vector|trace)",
        ],
    }

    # Natural language phrases -> preset names
    PRESET_ALIASES = {
        # Buttons
        "button": "button",
        "small button": "button_small",
        "medium button": "button",
        "large button": "button_large",
        "big button": "button_large",
        # Banners
        "banner": "banner",
        "twitter banner": "twitter_banner",
        "twitter header": "twitter_banner",
        "linkedin banner": "linkedin_banner",
        "linkedin cover": "linkedin_banner",
        "youtube banner": "youtube_banner",
        "facebook cover": "facebook_cover",
        "fb cover": "facebook_cover",
        # Social
        "instagram": "instagram_post",
        "instagram post": "instagram_post",
        "instagram square": "instagram_post",
        "instagram story": "instagram_story",
        "ig story": "instagram_story",
        "twitter post": "twitter_post",
        "tweet": "twitter_post",
        "facebook post": "facebook_post",
        "fb post": "facebook_post",
        "linkedin post": "linkedin_post",
        "youtube thumbnail": "youtube_thumbnail",
        "yt thumbnail": "youtube_thumbnail",
        # Print
        "postcard": "postcard",
        "flyer": "flyer",
        "poster": "poster",
        "business card": "business_card",
        "card": "business_card",
        # Web
        "icon": "icon",
        "favicon": "favicon",
        "thumbnail": "thumbnail",
        "thumb": "thumbnail",
        "hero": "hero",
        "hero image": "hero",
        "og image": "og_image",
        "social preview": "og_image",
        "avatar": "avatar",
        "profile": "avatar",
        "profile pic": "avatar",
    }

    def __init__(self):
        self.editor = PhotoEditor()
        self.converter = FormatConverter(output_base="output")

    def parse_and_execute(self, command: str, image_path: Optional[str] = None) -> bool:
        """Parse a natural language command and execute it.

        Returns True if the command was understood and succeeded.
        """
        command_lower = command.lower().strip()

        if image_path is None:
            image_path = self._extract_image_path(command)

        if image_path and not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            return False

        if self._matches_pattern(command_lower, "upscale"):
            return self._require_image(image_path, "upscale") \
                and self._do_upscale(image_path, command_lower)

        if self._matches_pattern(command_lower, "remove_bg"):
            return self._require_image(image_path, "remove background from") \
                and self._do_remove_bg(image_path)

        if self._matches_pattern(command_lower, "vectorize"):
            return self._require_image(image_path, "vectorize") \
                and self._do_vectorize(image_path)

        preset_name = self._extract_preset(command_lower)
        if preset_name:
            return self._require_image(image_path, f"convert to {preset_name}") \
                and self._do_convert(image_path, preset_name)

        print(f"Sorry, I didn't understand: {command}")
        print("\nTry commands like:")
        print("  - 'create a button from image.png'")
        print("  - 'make banner from photo.jpg'")
        print("  - 'upscale art.png 4x'")
        print("  - 'remove background from portrait.jpg'")
        print("  - 'vectorize logo.png'")
        return False

    # -- parsing --------------------------------------------------------------

    def _matches_pattern(self, text: str, pattern_type: str) -> bool:
        """Check whether the text matches any pattern of the given type."""
        return any(re.search(pattern, text) for pattern in self.PATTERNS[pattern_type])

    @staticmethod
    def _extract_image_path(command: str) -> Optional[str]:
        """Find an image path mentioned anywhere in the command."""
        for word in command.split():
            clean_word = word.strip("\"'.,;:")
            if clean_word.lower().endswith(IMAGE_EXTENSIONS) or os.path.exists(clean_word):
                return clean_word

        # "from X", "of X", "to X" phrasings
        for pattern in (r"from\s+[\"']?([^\s\"']+\.\w+)[\"']?",
                        r"of\s+[\"']?([^\s\"']+\.\w+)[\"']?",
                        r"to\s+[\"']?([^\s\"']+\.\w+)[\"']?"):
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_preset(self, command: str) -> Optional[str]:
        """Find a preset name or alias mentioned in the command."""
        for alias, preset_name in self.PRESET_ALIASES.items():
            if alias in command:
                return preset_name

        for preset_name in ALL_PRESETS:
            if preset_name.replace("_", " ") in command or preset_name in command:
                return preset_name

        return None

    @staticmethod
    def _require_image(image_path: Optional[str], action: str) -> bool:
        if not image_path:
            print(f"Please specify an image to {action}")
            return False
        return True

    # -- execution --------------------------------------------------------------

    @staticmethod
    def _report(result) -> bool:
        """Print a processing result and return whether it succeeded."""
        if result.success:
            print(f"Done! Saved to: {result.output_path}")
        else:
            print(result.message)
        return result.success

    @staticmethod
    def _output_path(folder: str, filename: str) -> Path:
        out_dir = Path("output") / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename

    def _do_upscale(self, image_path: str, command: str) -> bool:
        scale = 4
        match = re.search(r"(\d)x|(\d+)\s*times", command)
        if match:
            scale = int(match.group(1) or match.group(2))
            if scale not in (2, 4):
                scale = 4 if scale > 2 else 2
                print(f"Scale must be 2 or 4, using {scale}")

        print(f"Upscaling {image_path} by {scale}x...")
        input_path = Path(image_path)
        return self._report(self.editor.ai_upscale(
            input_path,
            self._output_path("upscaled", f"{input_path.stem}_upscaled_{scale}x.png"),
            scale=scale, model=UpscaleModel.from_name("general", scale),
        ))

    def _do_remove_bg(self, image_path: str) -> bool:
        print(f"Removing background from {image_path}...")
        input_path = Path(image_path)
        return self._report(self.editor.remove_background(
            input_path,
            self._output_path("no_background", f"{input_path.stem}_nobg.png"),
        ))

    def _do_vectorize(self, image_path: str) -> bool:
        print(f"Converting {image_path} to vector SVG...")
        input_path = Path(image_path)
        return self._report(self.editor.vectorize(
            input_path, self._output_path("vectors", f"{input_path.stem}.svg"),
        ))

    def _do_convert(self, image_path: str, preset_name: str) -> bool:
        preset = ALL_PRESETS.get(preset_name)
        if not preset:
            print(f"Unknown preset: {preset_name}")
            return False

        print(f"Converting {image_path} to {preset.name} ({preset.width}x{preset.height})...")
        return self._report(self.converter.convert(image_path, preset_name))


# ============================================================================
# Interactive mode
# ============================================================================

def interactive_mode() -> None:
    """Run a read-eval loop for natural language commands."""
    parser = CommandParser()

    print("""
╔══════════════════════════════════════════════════════════════╗
║          Smart Photo Editor - Interactive Mode               ║
╠══════════════════════════════════════════════════════════════╣
║  Just tell me what you want in plain English:                ║
║                                                              ║
║    > create a button from logo.png                           ║
║    > make banner from my_art.png                             ║
║    > upscale dragon.png 4x                                   ║
║    > remove background from photo.jpg                        ║
║    > vectorize icon.png                                      ║
║    > convert sunset.png to postcard                          ║
║                                                              ║
║  Commands: help, presets, quit                               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    while True:
        try:
            command = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not command:
            continue
        if command.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if command.lower() in ("help", "?"):
            print_help()
        elif command.lower() in ("presets", "formats", "list"):
            list_presets()
        else:
            parser.parse_and_execute(command)


def print_help() -> None:
    print("""
Available commands:

  FORMAT CONVERSION:
    create a button from image.png
    make banner from photo.jpg
    convert art.png to postcard
    turn logo.png into an icon

  UPSCALING (AI-powered):
    upscale image.png
    upscale art.png 4x
    make image.png bigger
    enlarge photo.jpg 2x

  BACKGROUND REMOVAL:
    remove background from photo.jpg
    make portrait.png transparent
    cut out subject from image.png

  VECTORIZATION:
    vectorize logo.png
    convert icon.png to svg
    make logo.png a vector

  OTHER:
    help     - Show this help
    presets  - List all format presets
    quit     - Exit interactive mode

Tips:
  - You can use natural language - just describe what you want!
  - Image paths can include spaces if quoted: "my image.png"
  - Output goes to ./output/ organized by type
""")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Natural language photo editor - just tell it what you want!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python smart_cli.py "create a button from logo.png"
    python smart_cli.py "upscale art.png 4x"
    python smart_cli.py "make a banner" image.png
    python smart_cli.py            # Interactive mode
        """,
    )
    parser.add_argument("command", nargs="?", help="Natural language command")
    parser.add_argument("image", nargs="?", help="Image path (optional if in command)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--presets", "-p", action="store_true", help="List all presets")
    args = parser.parse_args()

    if args.presets:
        list_presets()
    elif args.interactive or not args.command:
        interactive_mode()
    else:
        CommandParser().parse_and_execute(args.command, args.image)


if __name__ == "__main__":
    main()
