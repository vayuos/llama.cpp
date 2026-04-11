#!/bin/bash
set -e

# Gravitas Code: Native Build Script
# This script bundles and packages the extension for local installation.

echo "--- Building Gravitas Code (Custom Build) ---"

# 1. Clean previous build
rm -f gravitas-code-custom.vsix
rm -rf dist

# 2. Dependency Check
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# 3. Bundle with esbuild
echo "Bundling extension..."
npm run bundle

# 4. Package VSIX
echo "Packaging Extension..."
# Try to use local vsce if available, otherwise npx
if [ -f "./node_modules/.bin/vsce" ]; then
    ./node_modules/.bin/vsce package --out gravitas-code-custom.vsix --allow-star-activation --skip-license --allow-unused-files-pattern
else
    npx @vscode/vsce package --out gravitas-code-custom.vsix --allow-star-activation --skip-license --allow-unused-files-pattern
fi



echo "------------------------------------------------"
echo "BUILD SUCCESS: gravitas-code-custom.vsix"
echo "Install with: code --install-extension gravitas-code-custom.vsix --force"
echo "------------------------------------------------"
