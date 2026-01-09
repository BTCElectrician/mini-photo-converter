#!/usr/bin/env python3
"""
Smart CLI - Natural Language Photo Editor

Just tell it what you want in plain English:

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

Drag & drop mode (watches clipboard/current dir):
    python smart_cli.py --watch
"""

import os
import re
import sys
import readline  # For better input history
from pathlib import Path
from typing import Optional, Tuple, List

from photo_editor import PhotoEditor, UpscaleModel, VectorMode
from format_converter import FormatConverter
from presets import ALL_PRESETS, list_presets


# ============================================================================
# Natural Language Parser
# ============================================================================

class CommandParser:
    """Parse natural language commands into actions."""

    # Patterns for different commands
    PATTERNS = {
        'upscale': [
            r'(?:upscale|upsize|enlarge|make.+bigger|make.+larger|scale.+up|increase.+size)',
            r'(?:(\d)x|(\d+)\s*times)',  # Scale factor
        ],
        'remove_bg': [
            r'(?:remove.+background|remove.+bg|no.+background|transparent|cut.?out|extract)',
        ],
        'vectorize': [
            r'(?:vector|svg|vectorize|convert.+to.+svg|make.+vector|trace)',
        ],
        'convert': [
            r'(?:create|make|convert|turn.+into|as.+a|to.+a?)\s+(?:a\s+)?(\w+)',
        ],
    }

    # Preset aliases (natural language -> preset name)
    PRESET_ALIASES = {
        # Buttons
        'button': 'button',
        'small button': 'button_small',
        'medium button': 'button',
        'large button': 'button_large',
        'big button': 'button_large',

        # Banners
        'banner': 'banner',
        'twitter banner': 'twitter_banner',
        'twitter header': 'twitter_banner',
        'linkedin banner': 'linkedin_banner',
        'linkedin cover': 'linkedin_banner',
        'youtube banner': 'youtube_banner',
        'facebook cover': 'facebook_cover',
        'fb cover': 'facebook_cover',

        # Social
        'instagram': 'instagram_square',
        'instagram post': 'instagram_square',
        'instagram square': 'instagram_square',
        'instagram story': 'instagram_story',
        'ig story': 'instagram_story',
        'twitter post': 'twitter_post',
        'tweet': 'twitter_post',
        'facebook post': 'facebook_post',
        'fb post': 'facebook_post',
        'linkedin post': 'linkedin_post',
        'youtube thumbnail': 'youtube_thumbnail',
        'yt thumbnail': 'youtube_thumbnail',

        # Print
        'postcard': 'postcard',
        'flyer': 'flyer',
        'poster': 'poster',
        'business card': 'business_card',
        'card': 'business_card',

        # Web
        'icon': 'icon',
        'favicon': 'favicon',
        'thumbnail': 'thumbnail',
        'thumb': 'thumbnail',
        'hero': 'hero',
        'hero image': 'hero',
        'og image': 'og_image',
        'social preview': 'og_image',
        'avatar': 'avatar',
        'profile': 'avatar',
        'profile pic': 'avatar',
    }

    def __init__(self):
        self.editor = PhotoEditor()
        self.converter = FormatConverter(output_base="output")

    def parse_and_execute(self, command: str, image_path: Optional[str] = None) -> bool:
        """
        Parse a natural language command and execute it.

        Returns True if command was understood and executed.
        """
        command_lower = command.lower().strip()

        # Extract image path from command if not provided
        if image_path is None:
            image_path = self._extract_image_path(command)

        if image_path and not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            return False

        # Try to match command patterns

        # 1. Check for upscale
        if self._matches_pattern(command_lower, 'upscale'):
            if not image_path:
                print("Please specify an image to upscale")
                return False
            return self._do_upscale(image_path, command_lower)

        # 2. Check for background removal
        if self._matches_pattern(command_lower, 'remove_bg'):
            if not image_path:
                print("Please specify an image to remove background from")
                return False
            return self._do_remove_bg(image_path)

        # 3. Check for vectorize
        if self._matches_pattern(command_lower, 'vectorize'):
            if not image_path:
                print("Please specify an image to vectorize")
                return False
            return self._do_vectorize(image_path)

        # 4. Check for format conversion (banner, button, etc.)
        preset = self._extract_preset(command_lower)
        if preset:
            if not image_path:
                print(f"Please specify an image to convert to {preset}")
                return False
            return self._do_convert(image_path, preset)

        # 5. Check if they just gave us a preset name directly
        for alias, preset_name in self.PRESET_ALIASES.items():
            if alias in command_lower:
                if not image_path:
                    print(f"Please specify an image to convert to {preset_name}")
                    return False
                return self._do_convert(image_path, preset_name)

        # Couldn't understand
        print(f"Sorry, I didn't understand: {command}")
        print("\nTry commands like:")
        print("  - 'create a button from image.png'")
        print("  - 'make banner from photo.jpg'")
        print("  - 'upscale art.png 4x'")
        print("  - 'remove background from portrait.jpg'")
        print("  - 'vectorize logo.png'")
        return False

    def _matches_pattern(self, text: str, pattern_type: str) -> bool:
        """Check if text matches any pattern of the given type."""
        patterns = self.PATTERNS.get(pattern_type, [])
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _extract_image_path(self, command: str) -> Optional[str]:
        """Extract image path from command."""
        # Common image extensions
        extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif']

        # Try to find a file path in the command
        words = command.split()
        for word in words:
            # Remove quotes and common punctuation
            clean_word = word.strip('"\'.,;:')

            # Check if it looks like a file path
            for ext in extensions:
                if clean_word.lower().endswith(ext):
                    return clean_word

            # Check if adding an extension makes it a valid file
            if os.path.exists(clean_word):
                return clean_word

        # Check for "from X" or "of X" patterns
        patterns = [
            r'from\s+["\']?([^\s"\']+\.\w+)["\']?',
            r'of\s+["\']?([^\s"\']+\.\w+)["\']?',
            r'to\s+["\']?([^\s"\']+\.\w+)["\']?',
        ]
        for pattern in patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_preset(self, command: str) -> Optional[str]:
        """Extract preset name from command."""
        # First check aliases
        for alias, preset_name in self.PRESET_ALIASES.items():
            if alias in command:
                return preset_name

        # Then check actual preset names
        for preset_name in ALL_PRESETS.keys():
            if preset_name.replace('_', ' ') in command or preset_name in command:
                return preset_name

        return None

    def _do_upscale(self, image_path: str, command: str) -> bool:
        """Execute upscale command."""
        # Extract scale factor
        scale = 4  # Default
        match = re.search(r'(\d)x|(\d+)\s*times', command)
        if match:
            scale = int(match.group(1) or match.group(2))
            if scale not in [2, 4]:
                print(f"Scale must be 2 or 4, using {4 if scale > 2 else 2}")
                scale = 4 if scale > 2 else 2

        print(f"Upscaling {image_path} by {scale}x...")

        input_path = Path(image_path)
        output_dir = Path("output/upscaled")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_upscaled_{scale}x.png"

        model = UpscaleModel.GENERAL_X4 if scale == 4 else UpscaleModel.GENERAL_X2
        result = self.editor.ai_upscale(input_path, output_path, scale=scale, model=model)

        if result.success:
            print(f"Done! Saved to: {result.output_path}")
            return True
        else:
            print(f"Error: {result.message}")
            return False

    def _do_remove_bg(self, image_path: str) -> bool:
        """Execute background removal."""
        print(f"Removing background from {image_path}...")

        input_path = Path(image_path)
        output_dir = Path("output/no_background")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_nobg.png"

        result = self.editor.remove_background(input_path, output_path)

        if result.success:
            print(f"Done! Saved to: {result.output_path}")
            return True
        else:
            print(f"Error: {result.message}")
            return False

    def _do_vectorize(self, image_path: str) -> bool:
        """Execute vectorization."""
        print(f"Converting {image_path} to vector SVG...")

        input_path = Path(image_path)
        output_dir = Path("output/vectors")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.svg"

        result = self.editor.vectorize(input_path, output_path)

        if result.success:
            print(f"Done! Saved to: {result.output_path}")
            return True
        else:
            print(f"Error: {result.message}")
            return False

    def _do_convert(self, image_path: str, preset_name: str) -> bool:
        """Execute format conversion."""
        preset = ALL_PRESETS.get(preset_name)
        if not preset:
            print(f"Unknown preset: {preset_name}")
            return False

        print(f"Converting {image_path} to {preset.name} ({preset.width}x{preset.height})...")

        result = self.converter.convert(image_path, preset_name)

        if result.success:
            print(f"Done! Saved to: {result.output_path}")
            return True
        else:
            print(f"Error: {result.message}")
            return False


# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_mode():
    """Run in interactive chat mode."""
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

            if not command:
                continue

            # Handle special commands
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if command.lower() in ['help', '?']:
                print_help()
                continue

            if command.lower() in ['presets', 'formats', 'list']:
                list_presets()
                continue

            # Parse and execute
            parser.parse_and_execute(command)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


def print_help():
    """Print help message."""
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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Natural language photo editor - just tell it what you want!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Direct commands
    python smart_cli.py "create a button from logo.png"
    python smart_cli.py "upscale art.png 4x"
    python smart_cli.py "remove background from photo.jpg"

    # With separate image path
    python smart_cli.py "make a banner" image.png
    python smart_cli.py "upscale 4x" my_art.png

    # Interactive mode
    python smart_cli.py
        """
    )

    parser.add_argument("command", nargs="?", help="Natural language command")
    parser.add_argument("image", nargs="?", help="Image path (optional if included in command)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--presets", "-p", action="store_true", help="List all format presets")

    args = parser.parse_args()

    # List presets
    if args.presets:
        list_presets()
        return

    # Interactive mode
    if args.interactive or (not args.command and not args.image):
        interactive_mode()
        return

    # Direct command execution
    if args.command:
        cmd_parser = CommandParser()
        cmd_parser.parse_and_execute(args.command, args.image)


if __name__ == "__main__":
    main()
