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

# Exit on error
set -e

# Check if uv exists
if command -v uv &> /dev/null; then
    echo "uv is already installed"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Add uv to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Creating virtual environment..."
uv venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing requirements..."
uv pip install -r requirements.txt
uv pip install -r requirements-test.txt

echo "Running tests..."
pytest tests/ -v

echo "Running coverage report..."
pytest tests/ --cov=paprmcp --cov-report=term-missing

echo "Tests completed successfully!" 