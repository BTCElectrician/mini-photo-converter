"""
Drop Folder Watcher - Auto-process AI-generated images

This script watches a folder for new images and automatically processes them:
- Removes background
- Creates vector SVG
- Resizes to multiple sizes

Perfect for dumping AI-generated images and getting them production-ready!

Usage:
    python drop_watcher.py                    # Watch ./drop folder
    python drop_watcher.py --watch /my/folder # Watch custom folder
    python drop_watcher.py --no-vector        # Skip vectorization
    python drop_watcher.py --sizes 512 1024   # Custom output sizes

Folder Structure:
    drop/           <- Drop your images here
    processed/
        image1/
            image1_nobg.png    <- Background removed
            image1.svg         <- Vectorized
            image1_512.png     <- Resized versions
            image1_1024.png
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from photo_editor import PhotoEditor, VectorMode, ResizeMode, ProcessingResult


class ImageDropHandler(FileSystemEventHandler):
    """
    Handler for processing dropped images.

    Watches for new image files and processes them through the pipeline.
    """

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}

    def __init__(self,
                 output_dir: Path,
                 remove_bg: bool = True,
                 create_vector: bool = True,
                 resize_sizes: Optional[List[int]] = None,
                 vector_mode: VectorMode = VectorMode.ILLUSTRATION,
                 verbose: bool = True):
        """
        Initialize the drop handler.

        Args:
            output_dir: Directory for processed files
            remove_bg: Whether to remove background
            create_vector: Whether to create SVG
            resize_sizes: List of sizes for resizing (e.g., [512, 1024, 2048])
            vector_mode: Vectorization preset
            verbose: Print detailed progress
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.remove_bg = remove_bg
        self.create_vector = create_vector
        self.resize_sizes = resize_sizes or [512, 1024]
        self.vector_mode = vector_mode
        self.verbose = verbose
        self.editor = PhotoEditor()

        # Track processed files to avoid duplicates
        self._processed = set()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "   ", "OK": " + ", "FAIL": " X ", "WAIT": " ~ "}
        print(f"[{timestamp}] {prefix.get(level, '   ')} {message}")

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's a supported image format
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return

        # Skip already processed files
        if str(file_path) in self._processed:
            return

        # Wait a moment for file to finish writing
        self._wait_for_file(file_path)

        # Process the image
        self._process_image(file_path)

    def _wait_for_file(self, file_path: Path, timeout: int = 30):
        """Wait for file to finish being written."""
        self._log(f"New image detected: {file_path.name}", "WAIT")

        last_size = -1
        stable_count = 0

        for _ in range(timeout * 2):  # Check every 0.5 seconds
            try:
                current_size = file_path.stat().st_size
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 2:  # File size stable for 1 second
                        return
                else:
                    stable_count = 0
                last_size = current_size
            except OSError:
                pass
            time.sleep(0.5)

    def _process_image(self, file_path: Path):
        """Process a single image through the pipeline."""
        self._processed.add(str(file_path))

        # Create output directory for this image
        img_output_dir = self.output_dir / file_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Processing: {file_path.name}")

        results = []
        current_image = file_path

        # Step 1: Remove background
        if self.remove_bg:
            nobg_path = img_output_dir / f"{file_path.stem}_nobg.png"
            result = self.editor.remove_background(current_image, nobg_path)
            results.append(result)
            self._log_result(result)
            if result.success:
                current_image = Path(result.output_path)

        # Step 2: Vectorize
        if self.create_vector:
            svg_path = img_output_dir / f"{file_path.stem}.svg"
            result = self.editor.vectorize(current_image, svg_path, mode=self.vector_mode)
            results.append(result)
            self._log_result(result)

        # Step 3: Resize to multiple sizes
        for size in self.resize_sizes:
            resize_path = img_output_dir / f"{file_path.stem}_{size}.png"
            result = self.editor.smart_resize(
                current_image,
                resize_path,
                width=size,
                height=size,
                mode=ResizeMode.LANCZOS
            )
            results.append(result)
            self._log_result(result)

        # Summary
        success_count = sum(1 for r in results if r.success)
        self._log(f"Completed: {success_count}/{len(results)} operations successful")
        self._log(f"Output: {img_output_dir}")
        print()  # Blank line for readability

    def _log_result(self, result: ProcessingResult):
        """Log a processing result."""
        if result.success:
            size_info = ""
            if result.file_size_after:
                size_kb = result.file_size_after / 1024
                size_info = f" ({size_kb:.1f} KB)"
            self._log(f"{result.operation}: {Path(result.output_path).name}{size_info}", "OK")
        else:
            self._log(f"{result.operation}: {result.message}", "FAIL")


class DropFolderWatcher:
    """
    Watches a folder for new images and processes them automatically.

    Example:
        watcher = DropFolderWatcher("./drop", "./processed")
        watcher.start()  # Blocks and watches
    """

    def __init__(self,
                 watch_dir: str = "drop",
                 output_dir: str = "processed",
                 remove_bg: bool = True,
                 create_vector: bool = True,
                 resize_sizes: Optional[List[int]] = None,
                 vector_mode: VectorMode = VectorMode.ILLUSTRATION):
        """
        Initialize the folder watcher.

        Args:
            watch_dir: Directory to watch for new images
            output_dir: Directory for processed output
            remove_bg: Enable background removal
            create_vector: Enable SVG vectorization
            resize_sizes: List of output sizes
            vector_mode: Vectorization style preset
        """
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.remove_bg = remove_bg
        self.create_vector = create_vector
        self.resize_sizes = resize_sizes or [512, 1024]
        self.vector_mode = vector_mode

        # Create watch directory if needed
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        self.observer = Observer()
        self.handler = ImageDropHandler(
            output_dir=self.output_dir,
            remove_bg=self.remove_bg,
            create_vector=self.create_vector,
            resize_sizes=self.resize_sizes,
            vector_mode=self.vector_mode
        )

    def start(self, blocking: bool = True):
        """
        Start watching the folder.

        Args:
            blocking: If True, blocks until interrupted
        """
        print("=" * 60)
        print("  SuperCharged Photo Editor - Drop Folder Watcher")
        print("=" * 60)
        print()
        print(f"  Watching:      {self.watch_dir.absolute()}")
        print(f"  Output:        {self.output_dir.absolute()}")
        print(f"  Remove BG:     {'Yes' if self.remove_bg else 'No'}")
        print(f"  Vectorize:     {'Yes' if self.create_vector else 'No'}")
        print(f"  Resize to:     {', '.join(map(str, self.resize_sizes))}px")
        print()
        print("  Drop images into the watch folder to process them!")
        print("  Press Ctrl+C to stop.")
        print()
        print("-" * 60)

        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()

        if blocking:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nStopping watcher...")
                self.stop()

    def stop(self):
        """Stop watching."""
        self.observer.stop()
        self.observer.join()
        print("Watcher stopped.")

    def process_existing(self):
        """Process any existing images in the watch folder."""
        print("Checking for existing images...")
        count = 0
        for file_path in self.watch_dir.iterdir():
            if file_path.suffix.lower() in ImageDropHandler.SUPPORTED_FORMATS:
                self.handler._process_image(file_path)
                count += 1

        if count == 0:
            print("No existing images found.")
        else:
            print(f"Processed {count} existing images.")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Watch a folder and auto-process dropped images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python drop_watcher.py                        # Default: watch ./drop
    python drop_watcher.py --watch ~/Downloads    # Watch Downloads folder
    python drop_watcher.py --no-vector            # Skip vectorization
    python drop_watcher.py --sizes 256 512 1024   # Custom output sizes
    python drop_watcher.py --process-existing     # Process existing files first

The watcher will:
  1. Remove background (AI-powered)
  2. Create vector SVG
  3. Resize to multiple sizes (default: 512px, 1024px)
        """
    )

    parser.add_argument(
        "--watch", "-w",
        default="drop",
        help="Folder to watch for new images (default: ./drop)"
    )
    parser.add_argument(
        "--output", "-o",
        default="processed",
        help="Output folder for processed images (default: ./processed)"
    )
    parser.add_argument(
        "--no-bg-removal",
        action="store_true",
        help="Disable background removal"
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable SVG vectorization"
    )
    parser.add_argument(
        "--sizes", "-s",
        type=int,
        nargs="+",
        default=[512, 1024],
        help="Output sizes in pixels (default: 512 1024)"
    )
    parser.add_argument(
        "--vector-mode", "-m",
        choices=["photo", "illustration", "logo", "pixel_art"],
        default="illustration",
        help="Vectorization style (default: illustration)"
    )
    parser.add_argument(
        "--process-existing", "-p",
        action="store_true",
        help="Process existing images in watch folder before watching"
    )

    args = parser.parse_args()

    # Map vector mode string to enum
    vector_modes = {
        "photo": VectorMode.PHOTO,
        "illustration": VectorMode.ILLUSTRATION,
        "logo": VectorMode.LOGO,
        "pixel_art": VectorMode.PIXEL_ART
    }

    watcher = DropFolderWatcher(
        watch_dir=args.watch,
        output_dir=args.output,
        remove_bg=not args.no_bg_removal,
        create_vector=not args.no_vector,
        resize_sizes=args.sizes,
        vector_mode=vector_modes[args.vector_mode]
    )

    if args.process_existing:
        watcher.process_existing()

    watcher.start()


if __name__ == "__main__":
    main()
