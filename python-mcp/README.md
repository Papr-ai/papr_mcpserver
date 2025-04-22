# Papr MCP Server

A FastAPI-based MCP (Memory Control Protocol) server implementation for integrating with Papr's memory services (https://papr.ai).

## Prerequisites

- Python 3.10 or higher
- `uv` package manager (will be installed automatically by the setup script)

## Quick Start

1. Clone this repository:
```bash
git clone <repository-url>
cd python-mcp
```

2. Run the setup script:
```bash
python setup_mcp.py
```

The setup script will:
- Install `uv` if not already installed
- Create a virtual environment (you can specify a custom name or use the default '.venv')
- Install all required dependencies
- Configure your Papr API key (make sure you obtained one before running the script)
- Set up MCP configuration for your chosen client (Claude or Cursor)
- Optionally start the MCP server

## Setup Options

You can run the setup script with different options:

```bash
# Basic setup with all prompts
python3 setup_run_mcp.py

# Skip dependency installation
python3 setup_run_mcp.py --skip-deps

# Enable debug logging
python3 setup_run_mcp.py --debug

# Skip setup and run server directly
python3 setup_run_mcp.py --run-server
```

## Manual Server Start

If you chose not to start the server during setup, you can start it manually:

```bash
# Using setup script (recommended)
python3 setup_mcp.py --run-server

# Using uv directly
uv run python paprmcp.py



Note: Using the setup script with `--run-server` is recommended as it ensures the correct virtual environment is used and proper configuration is loaded.

## Configuration

The setup script creates two main configuration files:

1. `.env` file in the project root:
   - Contains your Papr API key
   - Sets the memory server URL (default is memory.papr.ai)

2. MCP configuration file (location depends on your OS and chosen client):
   - macOS: 
     - Claude: `~/Library/Application Support/claude/claude_desktop_config.json`
     - Cursor: `./cursor/mcp.json`
   - Windows:
     - Claude: `%APPDATA%/claude/claude_desktop_config.json`
     - Cursor: `./cursor/mcp.json`
   - Linux:
     - Claude: `~/.config/claude/claude_desktop_config.json`
     - Cursor: `./cursor/mcp.json`

## Development

The project uses `pyproject.toml` for dependency management with the following extras:
- `dev`: Development tools (debugpy, Flask, etc.)
- `test`: Testing tools (pytest, coverage, etc.)
- `all`: All of the above

To install specific extras:
```bash
uv pip install ".[dev]"  # Development dependencies
uv pip install ".[test]"  # Testing dependencies
uv pip install ".[all]"  # All dependencies
```

## Troubleshooting

If you encounter any issues:

1. Check the logs for detailed error messages
2. Ensure your Papr API key is correctly set in the `.env` file
3. Verify the virtual environment is activated
4. Make sure all dependencies are installed correctly

For additional help, please contact support or open an issue in the repository.

## Project Structure

```
python-mcp/
├── models/            # Data models and validators
├── services/          # Middleware services
├── tests/             # Test files
├── paprmcp.py         # Main MCP server file
├── requirements.txt   # Project dependencies
└── .env               # Environment configuration
```

Todo (stdio lcoal):

- use v1/search  for get_memory
- update mcp to read from fastapi  ()
- Tested Claude, test cursor and app in the setup the piece of code to append to any client 

Remote MCP Todo
-> add support for http route in MCP server (p0)
-> host in the cloud (p0)

