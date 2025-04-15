#!/bin/bash

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