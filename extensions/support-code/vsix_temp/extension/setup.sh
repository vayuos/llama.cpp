#!/bin/bash

# Configuration - All paths are now relative to the script's location
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SOURCE_DIR="$SCRIPT_DIR"
TARGET_DIR="$(readlink -f "$SCRIPT_DIR/../support-code")"
USER_NAME="viren"

# Check if run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

echo "Starting setup in $SOURCE_DIR..."

# Setup environment for the user (bypasses .bashrc non-interactive return)
SETUP_ENV="export NVM_DIR=\"/home/$USER_NAME/.nvm\"; [ -s \"\$NVM_DIR/nvm.sh\" ] && . \"\$NVM_DIR/nvm.sh\"; export BUN_INSTALL=\"/home/$USER_NAME/.bun\"; export PATH=\"\$BUN_INSTALL/bin:\$PATH\""

# 1. Check/Install Bun
echo "Checking for Bun..."
if ! sudo -u "$USER_NAME" bash -c "$SETUP_ENV; command -v bun &> /dev/null"; then
    echo "Bun not found. Installing Bun for $USER_NAME..."
    sudo -u "$USER_NAME" bash -c "curl -fsSL https://bun.sh/install | bash"
else
    echo "Bun is already installed."
fi

# 2. Install dependencies
cd "$SOURCE_DIR" || exit 1

# If node_modules already exists as a symlink or dir, handle it
if [ -L "node_modules" ]; then
    echo "Found existing node_modules symlink, removing to reinstall..."
    rm "node_modules"
elif [ -d "node_modules" ]; then
    echo "Found existing node_modules directory, removing to reinstall..."
    rm -rf "node_modules"
fi

echo "Installing dependencies as $USER_NAME..."
if sudo -u "$USER_NAME" bash -c "$SETUP_ENV; command -v bun &> /dev/null"; then
    echo "Using bun..."
    sudo -u "$USER_NAME" bash -c "$SETUP_ENV; cd \"$SOURCE_DIR\" && bun install"
else
    echo "Bun not found and couldn't be installed. Using npm..."
    sudo -u "$USER_NAME" bash -c "$SETUP_ENV; cd \"$SOURCE_DIR\" && npm install"
fi

# 3. Move to support-code
echo "Moving node_modules to $TARGET_DIR..."
# Ensure target dir exists
mkdir -p "$TARGET_DIR"

# Remove existing node_modules in target if any
if [ -d "$TARGET_DIR/node_modules" ]; then
    rm -rf "$TARGET_DIR/node_modules"
fi

mv "node_modules" "$TARGET_DIR/"

# 4. Create symlink (using relative path)
echo "Creating symbolic link..."
ln -s "../$(basename "$TARGET_DIR")/node_modules" "node_modules"

# 5. Fix ownership
chown -R "$USER_NAME:$USER_NAME" "$TARGET_DIR/node_modules"
chown -h "$USER_NAME:$USER_NAME" "node_modules"

echo "Setup complete! node_modules is now located in $TARGET_DIR/node_modules and linked to $(basename "$SOURCE_DIR")/node_modules"
