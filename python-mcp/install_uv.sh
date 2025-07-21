#!/bin/bash
# Copyright 2024 Papr AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

check_uv() {
    if command -v uv >/dev/null 2>&1; then
        echo "uv is already installed"
        return 0
    fi
    return 1
}

install_uv_mac() {
    echo "Installing uv on macOS..."
    if command -v brew >/dev/null 2>&1; then
        brew install uv
    else
        echo "Homebrew is not installed. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install uv
    fi
}

install_uv_ubuntu() {
    echo "Installing uv on Ubuntu..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
}

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! check_uv; then
        install_uv_mac
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Debian
    if ! check_uv; then
        install_uv_ubuntu
    fi
else
    echo "Unsupported operating system"
    exit 1
fi

# Verify installation
if check_uv; then
    echo "uv installation successful!"
    uv --version
else
    echo "Failed to install uv"
    exit 1
fi 