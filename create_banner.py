"""
Automatic Twitter Banner Creator

This script finds image files (.png, .jpg, .jpeg) in the current directory,
crops them centrally to a 3:1 aspect ratio, resizes them to 1500x500 pixels,
and saves the result with '_banner' appended to the original filename
into the 'processed-banner-photos' directory.

Usage:
    python create_banner.py

Example:
    Place 'my_photo.jpg' in the directory and run the script.
    It will create 'processed-banner-photos/my_photo_banner.jpg'.
"""

from PIL import Image
import sys
import os
import glob

# --- Configuration ---
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg')
OUTPUT_DIR = 'processed-banner-photos' # Directory to save banners

def create_twitter_banner(input_path, output_dir):
    """
    Loads an image, crops it centrally to a 3:1 aspect ratio, resizes to 1500x500,
    and saves it with '_banner' appended to the filename into the specified output directory.
    Returns True on success, False on failure.
    """
    base_filename = os.path.basename(input_path)
    try:
        img = Image.open(input_path)
        print(f"Processing: {base_filename} (Size: {img.width}x{img.height})")
    except FileNotFoundError:
        print(f"Error: Skipping {input_path} - File not found during processing loop (should not happen).")
        return False
    except Exception as e:
        print(f"Error opening image {input_path}: {e}. Skipping.")
        return False

    original_width, original_height = img.size
    target_width = 1500
    target_height = 500
    target_aspect_ratio = target_width / target_height # Should be 3.0

    # Calculate the aspect ratio of the original image
    original_aspect_ratio = original_width / original_height

    # Determine cropping box
    if original_aspect_ratio > target_aspect_ratio:
        new_width = int(target_aspect_ratio * original_height)
        offset = (original_width - new_width) / 2
        crop_box = (offset, 0, original_width - offset, original_height)
    elif original_aspect_ratio < target_aspect_ratio:
        new_height = int(original_width / target_aspect_ratio)
        offset = (original_height - new_height) / 2
        crop_box = (0, offset, original_width, original_height - offset)
    else:
        crop_box = (0, 0, original_width, original_height)

    # Generate output path inside the specified directory
    base, ext = os.path.splitext(base_filename)
    output_filename = f"{base}_banner{ext}"
    output_path = os.path.join(output_dir, output_filename)

    # Process and save
    try:
        img_cropped = img.crop(crop_box)
        img_resized = img_cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)
        print(f"  Successfully saved banner to: {output_path}")
        return True
    except Exception as e:
        print(f"  Error during processing or saving {base_filename}: {e}. Skipping.")
        return False

if __name__ == "__main__":
    print("Starting Automatic Twitter Banner Creation...")

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

    # Filter out any potential duplicates and files that might be banners already
    unique_files = sorted(list(set(image_files)))
    files_to_process = [f for f in unique_files if '_banner' not in os.path.splitext(f)[0] and os.path.isfile(f)]

    if not files_to_process:
        print("No suitable image files (.png, .jpg, .jpeg) found in the current directory to process.")
    else:
        print(f"Found {len(files_to_process)} images to process.")
        processed_count = 0
        for file_path in files_to_process:
            # Pass the output directory to the function
            if create_twitter_banner(file_path, OUTPUT_DIR):
                processed_count += 1

        print(f"\nFinished processing.")
        print(f"Successfully created {processed_count} banners in '{OUTPUT_DIR}'.")
        if processed_count < len(files_to_process):
            print(f"{len(files_to_process) - processed_count} images encountered errors.") 