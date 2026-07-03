"""
Drop Folder Watcher - auto-process images as they land in a folder.

Watches a folder for new images and runs each one through the pipeline:
background removal, SVG vectorization, and resizing to multiple sizes.
Perfect for dumping AI-generated images and getting them production-ready.

Usage:
    python drop_watcher.py                    # Watch ./drop
    python drop_watcher.py --watch /my/folder
    python drop_watcher.py --no-vector
    python drop_watcher.py --sizes 512 1024

Output layout:
    drop/                      <- Drop images here
    processed/<name>/
        <name>_nobg.png        <- Background removed
        <name>.svg             <- Vectorized
        <name>_512.png         <- Resized versions
        <name>_1024.png
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from photo_editor import PhotoEditor, ProcessingResult, ResizeMode, VectorMode

VECTOR_MODES = {mode.value: mode for mode in VectorMode}


class ImageDropHandler(FileSystemEventHandler):
    """Process new image files through the editing pipeline."""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self, output_dir: Path,
                 remove_bg: bool = True,
                 create_vector: bool = True,
                 resize_sizes: Optional[List[int]] = None,
                 vector_mode: VectorMode = VectorMode.ILLUSTRATION,
                 verbose: bool = True):
        """
        Args:
            output_dir: Directory for processed files.
            remove_bg: Remove the background.
            create_vector: Produce an SVG.
            resize_sizes: Output sizes in pixels (default: [512, 1024]).
            vector_mode: Vectorization preset.
            verbose: Print detailed progress.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.remove_bg = remove_bg
        self.create_vector = create_vector
        self.resize_sizes = resize_sizes or [512, 1024]
        self.vector_mode = vector_mode
        self.verbose = verbose
        self.editor = PhotoEditor()
        self._processed: set[str] = set()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_created(self, event):
        """Process newly created image files."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return
        if str(file_path) in self._processed:
            return

        self._wait_for_file(file_path)
        self.process(file_path)

    def process(self, file_path: Path) -> None:
        """Run one image through the pipeline and log each step."""
        self._processed.add(str(file_path))

        img_output_dir = self.output_dir / file_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Processing: {file_path.name}")

        results: list[ProcessingResult] = []
        current = file_path

        if self.remove_bg:
            result = self.editor.remove_background(
                current, img_output_dir / f"{file_path.stem}_nobg.png")
            results.append(result)
            self._log_result(result)
            if result.success:
                current = Path(result.output_path)

        if self.create_vector:
            result = self.editor.vectorize(
                current, img_output_dir / f"{file_path.stem}.svg",
                mode=self.vector_mode)
            results.append(result)
            self._log_result(result)

        for size in self.resize_sizes:
            result = self.editor.smart_resize(
                current, img_output_dir / f"{file_path.stem}_{size}.png",
                width=size, height=size, mode=ResizeMode.LANCZOS)
            results.append(result)
            self._log_result(result)

        success_count = sum(r.success for r in results)
        self._log(f"Completed: {success_count}/{len(results)} operations successful")
        self._log(f"Output: {img_output_dir}")
        print()

    def _wait_for_file(self, file_path: Path, timeout: int = 30) -> None:
        """Wait until the file size is stable, i.e. writing has finished."""
        self._log(f"New image detected: {file_path.name}", "WAIT")

        last_size = -1
        stable_count = 0
        for _ in range(timeout * 2):  # Poll every 0.5 seconds
            try:
                current_size = file_path.stat().st_size
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 2:  # Stable for a full second
                        return
                else:
                    stable_count = 0
                last_size = current_size
            except OSError:
                pass
            time.sleep(0.5)

    def _log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "   ", "OK": " + ", "FAIL": " X ", "WAIT": " ~ "}
        print(f"[{timestamp}] {prefix.get(level, '   ')} {message}")

    def _log_result(self, result: ProcessingResult) -> None:
        if result.success:
            size_info = ""
            if result.file_size_after:
                size_info = f" ({result.file_size_after / 1024:.1f} KB)"
            self._log(f"{result.operation}: {Path(result.output_path).name}{size_info}", "OK")
        else:
            self._log(f"{result.operation}: {result.message}", "FAIL")


class DropFolderWatcher:
    """Watch a folder and process every image dropped into it.

    Example:
        watcher = DropFolderWatcher("./drop", "./processed")
        watcher.start()  # Blocks until interrupted
    """

    def __init__(self, watch_dir: str = "drop",
                 output_dir: str = "processed",
                 remove_bg: bool = True,
                 create_vector: bool = True,
                 resize_sizes: Optional[List[int]] = None,
                 vector_mode: VectorMode = VectorMode.ILLUSTRATION):
        """
        Args:
            watch_dir: Directory to watch for new images.
            output_dir: Directory for processed output.
            remove_bg: Enable background removal.
            create_vector: Enable SVG vectorization.
            resize_sizes: Output sizes in pixels.
            vector_mode: Vectorization style preset.
        """
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        self.observer = Observer()
        self.handler = ImageDropHandler(
            output_dir=Path(output_dir),
            remove_bg=remove_bg,
            create_vector=create_vector,
            resize_sizes=resize_sizes,
            vector_mode=vector_mode,
        )

    def start(self, blocking: bool = True) -> None:
        """Start watching; blocks until Ctrl+C unless ``blocking=False``."""
        handler = self.handler
        print("=" * 60)
        print("  Photo Editor - Drop Folder Watcher")
        print("=" * 60)
        print()
        print(f"  Watching:      {self.watch_dir.absolute()}")
        print(f"  Output:        {handler.output_dir.absolute()}")
        print(f"  Remove BG:     {'Yes' if handler.remove_bg else 'No'}")
        print(f"  Vectorize:     {'Yes' if handler.create_vector else 'No'}")
        print(f"  Resize to:     {', '.join(map(str, handler.resize_sizes))}px")
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

    def stop(self) -> None:
        """Stop watching."""
        self.observer.stop()
        self.observer.join()
        print("Watcher stopped.")

    def process_existing(self) -> None:
        """Process images already sitting in the watch folder."""
        print("Checking for existing images...")
        count = 0
        for file_path in self.watch_dir.iterdir():
            if file_path.suffix.lower() in ImageDropHandler.SUPPORTED_FORMATS:
                self.handler.process(file_path)
                count += 1

        print(f"Processed {count} existing images." if count else "No existing images found.")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a folder and auto-process dropped images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python drop_watcher.py                        # Default: watch ./drop
    python drop_watcher.py --watch ~/Downloads
    python drop_watcher.py --no-vector
    python drop_watcher.py --sizes 256 512 1024
    python drop_watcher.py --process-existing

The watcher will:
  1. Remove background (AI-powered)
  2. Create vector SVG
  3. Resize to multiple sizes (default: 512px, 1024px)
        """,
    )
    parser.add_argument("--watch", "-w", default="drop",
                        help="Folder to watch for new images (default: ./drop)")
    parser.add_argument("--output", "-o", default="processed",
                        help="Output folder for processed images (default: ./processed)")
    parser.add_argument("--no-bg-removal", action="store_true",
                        help="Disable background removal")
    parser.add_argument("--no-vector", action="store_true",
                        help="Disable SVG vectorization")
    parser.add_argument("--sizes", "-s", type=int, nargs="+", default=[512, 1024],
                        help="Output sizes in pixels (default: 512 1024)")
    parser.add_argument("--vector-mode", "-m", choices=sorted(VECTOR_MODES),
                        default="illustration",
                        help="Vectorization style (default: illustration)")
    parser.add_argument("--process-existing", "-p", action="store_true",
                        help="Process existing images before watching")
    args = parser.parse_args()

    watcher = DropFolderWatcher(
        watch_dir=args.watch,
        output_dir=args.output,
        remove_bg=not args.no_bg_removal,
        create_vector=not args.no_vector,
        resize_sizes=args.sizes,
        vector_mode=VECTOR_MODES[args.vector_mode],
    )

    if args.process_existing:
        watcher.process_existing()

    watcher.start()


if __name__ == "__main__":
    main()
