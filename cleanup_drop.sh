#!/bin/bash
# Cleanup script: Archives processed images from drop/ to archive/
# Usage: ./cleanup_drop.sh [image_name]  (if no name, archives all)

ARCHIVE_DIR="archive"
DROP_DIR="drop"

# Create archive if it doesn't exist
mkdir -p "$ARCHIVE_DIR"

if [ -z "$1" ]; then
    # Archive all images in drop (except README)
    echo "Archiving all images from drop/ to archive/..."
    find "$DROP_DIR" -maxdepth 1 -type f \( \
        -iname "*.jpg" -o \
        -iname "*.jpeg" -o \
        -iname "*.png" \
    \) -exec mv {} "$ARCHIVE_DIR/" \;
    
    # Show what was archived
    if [ "$(ls -A "$ARCHIVE_DIR" 2>/dev/null)" ]; then
        echo "  ✓ Archived images:"
        ls -lh "$ARCHIVE_DIR" | tail -n +2 | awk '{print "    " $9}'
    fi
else
    # Archive specific image
    if [ -f "$DROP_DIR/$1" ]; then
        mv "$DROP_DIR/$1" "$ARCHIVE_DIR/$1"
        echo "✓ Archived: $1"
    else
        echo "✗ Not found: $DROP_DIR/$1"
        exit 1
    fi
fi

echo ""
echo "Drop folder now contains:"
ls -lh "$DROP_DIR" | grep -v README || echo "  (empty - ready for next image)"
