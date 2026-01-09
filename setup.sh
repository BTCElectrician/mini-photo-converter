#!/bin/bash
# Setup script for Photo Editor CLI
# Adds 'photo' command to your PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Photo Editor CLI Setup"
echo "======================"
echo ""

# Check if already in PATH
if command -v photo &> /dev/null; then
    echo "photo command is already available!"
    photo help
    exit 0
fi

# Determine shell config file
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    if [[ -f "$HOME/.bash_profile" ]]; then
        SHELL_RC="$HOME/.bash_profile"
    else
        SHELL_RC="$HOME/.bashrc"
    fi
else
    SHELL_RC="$HOME/.profile"
fi

echo "Adding to PATH in $SHELL_RC"
echo ""

# Add to PATH
echo "" >> "$SHELL_RC"
echo "# Photo Editor CLI" >> "$SHELL_RC"
echo "export PATH=\"\$PATH:$SCRIPT_DIR\"" >> "$SHELL_RC"

echo "Done! Run this to activate now:"
echo ""
echo "    source $SHELL_RC"
echo ""
echo "Or just use the full path:"
echo ""
echo "    $SCRIPT_DIR/photo banner image.png"
echo ""
echo "Quick commands:"
echo "    photo banner image.png    - Create a banner"
echo "    photo button image.png    - Create a button"
echo "    photo upscale image.png   - AI upscale 4x"
echo "    photo rembg image.png     - Remove background"
echo "    photo list                - Show all presets"
