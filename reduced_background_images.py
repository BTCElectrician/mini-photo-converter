import os
from PIL import Image
import io

# Configuration
TARGET_SIZE_KB = 200
TARGET_SIZE_BYTES = TARGET_SIZE_KB * 1024
SOURCE_DIR = '.' # Current directory where the script is run
OUTPUT_DIR = 'reduced_background_images'
SUFFIX = '_reduced'

def optimize_and_save_image(img_path, base_output_name):
    """Optimizes an image and saves it if under the target size.
    Tries PNG optimization first, then WebP lossless.
    Returns the path of the saved file or None if not saved.
    """
    try:
        img = Image.open(img_path)
        original_size = os.path.getsize(img_path)
        print(f"\nProcessing {os.path.basename(img_path)}...")
        print(f"  Original size: {original_size / 1024:.2f} KB")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # --- Try Optimized PNG ---
        buffer_png = io.BytesIO()
        # optimize=True enables various PNG optimization filters/strategies
        # compress_level=9 uses the highest zlib compression
        img.save(buffer_png, format='PNG', optimize=True, compress_level=9)
        buffer_png.seek(0)
        optimized_png_size = len(buffer_png.getvalue())
        print(f"  Optimized PNG size: {optimized_png_size / 1024:.2f} KB")

        if optimized_png_size <= TARGET_SIZE_BYTES:
            output_path_png = os.path.join(OUTPUT_DIR, f"{base_output_name}.png")
            with open(output_path_png, 'wb') as f:
                f.write(buffer_png.getvalue())
            print(f"  Saved optimized PNG to {output_path_png}")
            return output_path_png
        else:
            print(f"  Optimized PNG ({optimized_png_size / 1024:.2f} KB) > Target ({TARGET_SIZE_KB} KB)")

            # --- Try WebP Lossless ---
            buffer_webp = io.BytesIO()
            # Save using lossless WebP format
            img.save(buffer_webp, format='WEBP', lossless=True)
            buffer_webp.seek(0)
            webp_lossless_size = len(buffer_webp.getvalue())
            print(f"  Trying WebP (lossless) size: {webp_lossless_size / 1024:.2f} KB")

            if webp_lossless_size <= TARGET_SIZE_BYTES:
                output_path_webp = os.path.join(OUTPUT_DIR, f"{base_output_name}.webp")
                with open(output_path_webp, 'wb') as f:
                    f.write(buffer_webp.getvalue())
                print(f"  Saved as WebP (lossless) to {output_path_webp}")
                return output_path_webp
            else:
                print(f"  WebP (lossless) ({webp_lossless_size / 1024:.2f} KB) > Target ({TARGET_SIZE_KB} KB).")

                # --- Try WebP Lossy ---
                WEBP_QUALITY = 85 # Quality setting (0-100)
                print(f"  Trying WebP (lossy, quality={WEBP_QUALITY})...")
                buffer_webp_lossy = io.BytesIO()
                # Save using lossy WebP format
                img.save(buffer_webp_lossy, format='WEBP', quality=WEBP_QUALITY)
                buffer_webp_lossy.seek(0)
                webp_lossy_size = len(buffer_webp_lossy.getvalue())
                print(f"  WebP (lossy) size: {webp_lossy_size / 1024:.2f} KB")

                if webp_lossy_size <= TARGET_SIZE_BYTES:
                    output_path_webp_lossy = os.path.join(OUTPUT_DIR, f"{base_output_name}.webp") # Still save as .webp
                    with open(output_path_webp_lossy, 'wb') as f:
                        f.write(buffer_webp_lossy.getvalue())
                    print(f"  Saved as WebP (lossy, quality={WEBP_QUALITY}) to {output_path_webp_lossy}")
                    return output_path_webp_lossy
                else:
                    print(f"  WebP (lossy) ({webp_lossy_size / 1024:.2f} KB) > Target ({TARGET_SIZE_KB} KB). Skipping.")
                    return None

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_images():
    """Processes all PNG images in the source directory."""
    processed_files = []
    skipped_files = []

    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    print(f"Scanning for PNG files in '{os.path.abspath(SOURCE_DIR)}'...")
    found_png = False
    for filename in os.listdir(SOURCE_DIR):
        # Ensure we only process files directly in the source dir, not subdirs
        source_path = os.path.join(SOURCE_DIR, filename)
        if filename.lower().endswith('.png') and os.path.isfile(source_path):
            found_png = True
            base, _ = os.path.splitext(filename)
            base_output_name = f"{base}{SUFFIX}"

            saved_path = optimize_and_save_image(source_path, base_output_name)
            if saved_path:
                processed_files.append(os.path.basename(saved_path))
            else:
                skipped_files.append(filename)

    if not found_png:
         print("No PNG files found in the current directory.")
         return

    print("\n--- Processing Summary ---")
    if processed_files:
        print(f"Successfully processed ({len(processed_files)}):")
        for f in processed_files: print(f"  - {f}")
    if skipped_files:
        print(f"Skipped (still too large after optimization) ({len(skipped_files)}):")
        for f in skipped_files: print(f"  - {f}")
    print("-------------------------")


if __name__ == "__main__":
    # Check for Pillow installation at runtime
    try:
        from PIL import Image
    except ImportError:
        print("\nError: The 'Pillow' library is required but not installed.")
        print("You can install it by running this command in your terminal:")
        print("pip install Pillow")
        print("-------------------------")
        exit(1)

    process_images() 