"""
Automatic Letterbox Twitter Banner Creator

This script finds image files (.png, .jpg, .jpeg) in the current directory,
resizes them to fit a 1500px width while maintaining aspect ratio,
pastes them onto a 1500x500 black background (letterboxing),
and saves the result with '_banner_lb' appended to the original filename
into the 'letterbox-banners' directory.

Usage:
    python create_letterbox_banner.py

Example:
    Place 'my_wide_photo.jpg' in the directory and run the script.
    It will create 'letterbox-banners/my_wide_photo_banner_lb.jpg'.
"""

from PIL import Image, ImageOps
import sys
import os
import glob

# --- Configuration ---
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg')
OUTPUT_DIR = 'letterbox-banners'  # Directory to save letterboxed banners
TARGET_WIDTH = 1500
TARGET_HEIGHT = 500
BACKGROUND_COLOR = (0, 0, 0)  # Black background for letterboxing

def create_letterbox_banner(input_path, output_dir):
    """
    Loads an image, resizes it to fit 1500px width maintaining aspect ratio,
    pastes it onto a 1500x500 black background, and saves it into the
    specified output directory with '_banner_lb' appended.
    Returns True on success, False on failure.
    """
    base_filename = os.path.basename(input_path)
    try:
        img = Image.open(input_path)
        # Convert palette images (like GIFs) or single-channel images to RGB
        if img.mode == 'P' or img.mode == 'L':
            img = img.convert('RGB')
        # Ensure image has 3 channels (RGB) for consistency with background
        elif img.mode == 'RGBA':
             # If RGBA, create an RGB background and paste RGBA image onto it
             # This handles transparency correctly against the black background
            rgb_img = Image.new("RGB", img.size, BACKGROUND_COLOR)
            rgb_img.paste(img, mask=img.split()[3]) # Paste using alpha channel as mask
            img = rgb_img
        elif img.mode != 'RGB':
             img = img.convert('RGB') # Attempt conversion for other modes

        print(f"Processing: {base_filename} (Size: {img.width}x{img.height})")

    except FileNotFoundError:
        print(f"Error: Skipping {input_path} - File not found.")
        return False
    except Exception as e:
        print(f"Error opening or converting image {input_path}: {e}. Skipping.")
        return False

    # --- Resize with aspect ratio preservation ---
    img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    resized_width, resized_height = img.size
    print(f"  Resized to fit: {resized_width}x{resized_height}")

    # --- Create background canvas ---
    background = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), BACKGROUND_COLOR)

    # --- Calculate pasting position (center) ---
    paste_x = (TARGET_WIDTH - resized_width) // 2
    paste_y = (TARGET_HEIGHT - resized_height) // 2
    paste_position = (paste_x, paste_y)

    # --- Paste onto background ---
    background.paste(img, paste_position)

    # --- Generate output path ---
    base, ext = os.path.splitext(base_filename)
    output_filename = f"{base}_banner_lb{ext}"
    output_path = os.path.join(output_dir, output_filename)

    # --- Save the final banner ---
    try:
        background.save(output_path)
        print(f"  Successfully saved letterbox banner to: {output_path}")
        return True
    except Exception as e:
        print(f"  Error saving banner {output_filename}: {e}. Skipping.")
        return False

if __name__ == "__main__":
    print("Starting Automatic Letterbox Twitter Banner Creation...")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"Created directory: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error creating directory {OUTPUT_DIR}: {e}. Exiting.")
            sys.exit(1)

    image_files = []
    # Find all supported image files in the current directory
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(glob.glob(f'*{ext}'))
        image_files.extend(glob.glob(f'*{ext.upper()}')) # Include uppercase extensions

    # Filter out duplicates and files that might be banners already
    unique_files = sorted(list(set(image_files)))
    files_to_process = [f for f in unique_files if '_banner_lb' not in os.path.splitext(f)[0] and os.path.isfile(f)]

    if not files_to_process:
        print("No suitable image files (.png, .jpg, .jpeg) found in the current directory to process.")
    else:
        print(f"Found {len(files_to_process)} images to process.")
        processed_count = 0
        for file_path in files_to_process:
            if create_letterbox_banner(file_path, OUTPUT_DIR):
                processed_count += 1

        print(f"\nFinished processing.")
        print(f"Successfully created {processed_count} letterbox banners in '{OUTPUT_DIR}'.")
        if processed_count < len(files_to_process):
            print(f"{len(files_to_process) - processed_count} images encountered errors.") 