#!/bin/bash

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*|MINGW32*|MSYS*|MINGW*) echo "windows";;
        *)          echo "unknown";;
    esac
}

# Get OS type
OS_TYPE=$(detect_os)

# Run appropriate script based on OS
case "$OS_TYPE" in
    "linux")
        echo "Running Linux installation and tests..."
        ./scripts/install_and_test_linux.sh
        ;;
    "macos")
        echo "Running macOS installation and tests..."
        ./scripts/install_and_test_macos.sh
        ;;
    "windows")
        echo "Running Windows installation and tests..."
        ./scripts/install_and_test_windows.bat
        ;;
    *)
        echo "Unsupported operating system"
        exit 1
        ;;
esac 