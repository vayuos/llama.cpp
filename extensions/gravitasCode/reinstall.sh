#!/bin/bash
set -e

echo "🔄 Reinstalling Gravitas Code Extension..."
echo ""

echo ""

# Uninstall old version
echo "1️⃣ Uninstalling old version..."
code --uninstall-extension vayuos.gravitas-code || true

# Install new version
echo "2️⃣ Installing new version..."
code --install-extension gravitas-code-0.1.0.vsix --force

echo ""
echo "✅ Installation complete!"
echo ""
echo "3️⃣ Restarting VS Code..."

# Kill all VS Code processes
pkill -f "code.*gravitas-code" || true
sleep 1
pkill -9 code || true
sleep 2

# Reopen VS Code in the current directory
echo "4️⃣ Reopening VS Code..."
code . &

echo ""
echo "✅ VS Code restarted with new extension!"
echo "   Open Setup Wizard to see the changes."
echo ""
