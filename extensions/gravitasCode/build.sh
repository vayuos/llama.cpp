#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SUPPORT_DIR="$(readlink -f "$SCRIPT_DIR/../support-code")"
VSIX_TEMP="$SUPPORT_DIR/vsix_temp"
BUILD_DIR="/tmp/vsix_build"
OUTPUT="$SCRIPT_DIR/gravitas-code-custom.vsix"

echo "🔨 Building Gravitas Code Extension..."

# 1. Source NVM if available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# 2. Bundle
echo "📦 Bundling with esbuild..."
cd "$SCRIPT_DIR"
npm run bundle

# 3. Package .vsix
echo "📋 Packaging .vsix..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/extension"
cp -r dist media img package.json README.md setup.sh .gravitas "$BUILD_DIR/extension/"
cp "$VSIX_TEMP/extension.vsixmanifest" "$BUILD_DIR/"
cp "$VSIX_TEMP/[Content_Types].xml" "$BUILD_DIR/"
cd "$BUILD_DIR"
zip -r "$OUTPUT" .

echo ""
echo "✅ Built: $OUTPUT"
echo "   Size: $(du -h "$OUTPUT" | cut -f1)"
