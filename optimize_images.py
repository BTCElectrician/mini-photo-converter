"""
Higher Quality Image Optimizer and Organizer

This script optimizes PNG images found in the current directory:
1. Resizes them to a larger 256x256 pixels.
2. Uses less aggressive compression (quality=95) for clarity.
3. Saves optimized versions to the 'optimized-hq' folder.
4. Moves the original processed images to the 'processed-photos' folder.

Usage:
1. Place PNG images in the same directory as this script.
2. Run this script.
3. Find optimized images in 'optimized-hq'.
4. Find original images moved to 'processed-photos'.
"""

from PIL import Image
import os
import glob
import shutil # Import shutil for moving files

# --- Directory Setup ---
output_dir = 'optimized-hq'
processed_dir = 'processed-photos' # Directory for original processed files

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Create processed photos directory if it doesn't exist
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    print(f"Created directory: {processed_dir}")

# --- Find and Process Images ---

# Find all PNG files in the current directory (excluding subdirs)
image_files = [f for f in glob.glob('*.png') if os.path.isfile(f)]

if not image_files:
    print("No PNG files found in the current directory to process.")
else:
    print(f"Found {len(image_files)} PNG files to process.")

processed_count = 0
# Resize, optimize, and move each image
for img_path in image_files:
    try:
        # Construct paths
        base_filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, base_filename)
        processed_path = os.path.join(processed_dir, base_filename)

        # Open image
        img = Image.open(img_path)

        # Get original size for reporting
        original_size = os.path.getsize(img_path) / (1024 * 1024)  # Size in MB

        # Resize to 256x256 - larger size for better quality
        img.thumbnail((256, 256))

        # Save optimized image with higher quality
        img.save(output_path, optimize=True, quality=95)  # Higher quality setting

        # Get new size for reporting
        new_size = os.path.getsize(output_path) / 1024  # Size in KB

        print(f"Optimized: {base_filename} ({original_size:.2f} MB) → {output_path} ({new_size:.2f} KB)")

        # Move the original file to the processed directory
        shutil.move(img_path, processed_path)
        print(f"Moved original: {base_filename} → {processed_path}")
        processed_count += 1

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        # Optionally, decide if you want to move the file even if optimization failed
        # shutil.move(img_path, processed_path)

# --- Final Summary ---
if processed_count > 0:
    print(f"\nDone! {processed_count} images have been optimized and saved to '{output_dir}'.")
    print(f"Original images have been moved to '{processed_dir}'.")
elif not image_files:
    pass # Message already printed if no files were found
else:
    print("\nNo images were successfully processed.") 