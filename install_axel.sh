#!/bin/bash

set -e

echo "🚀 Installing Axel (Download Accelerator)..."

# Detect package manager
if command -v apt &>/dev/null; then
    echo "Detected apt (Debian/Ubuntu)"
    sudo apt update -y
    sudo apt install -y axel
elif command -v dnf &>/dev/null; then
    echo "Detected dnf (Fedora/RHEL)"
    sudo dnf install -y axel
elif command -v yum &>/dev/null; then
    echo "Detected yum (CentOS)"
    sudo yum install -y axel
elif command -v pacman &>/dev/null; then
    echo "Detected pacman (Arch)"
    sudo pacman -Sy --noconfirm axel
elif command -v zypper &>/dev/null; then
    echo "Detected zypper (openSUSE)"
    sudo zypper install -y axel
else
    echo "❌ Unsupported distribution. Please install Axel manually."
    exit 1
fi

echo "✅ Axel installed successfully!"
axel --version
