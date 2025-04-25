# Papr MCP Server

A FastAPI-based MCP (Memory Control Protocol) server implementation for integrating with Papr's memory services (https://papr.ai).

## Prerequisites

- Python 3.10 or higher
- **Get your API key:** You can find it in the settings section of **[papr.ai](https://papr.ai)**. You'll need to create an account first and quickly go through our web app onboarding.

## Quick Start

1. Clone this repository for Python MCP Papr Server:
```bash
git clone https://github.com/Papr-ai/papr_mcpserver
cd python-mcp
```

2. Run the setup script:
```bash
python3 setup_run_mcp.py
```

The setup script will guide you through the following steps:

1. **Dependencies Installation**
   - Installs `uv` if not already present
   - Creates a virtual environment (default '.venv')
   - Installs all required project dependencies

2. **API Key Configuration**
   - Prompts for your Papr API key
   - Validates the key format
   - Stores it securely in `.env` file

3. **MCP Client Selection**
   - Choose your preferred client:
     - Claude
     - Cursor AI
     - Other

4. **Client Configuration**
   - For Claude: Automatically configures `claude_desktop_config.json`
   - For Cursor AI: Automatically configures `./cursor/mcp.json`
   - For Other clients: Displays the configuration JSON in console (you'll need to manually copy this to your client's configuration)

> **Note:** If you select "Other" as your client, the script will print the MCP configuration JSON to the console. You'll need to manually copy this configuration to your client's appropriate configuration file.

The script will then:
- Start the MCP server (optional)


> **Tip:** You can always start the server later using the options described in the "Manual Server Start" section below.

## Setup Options

You can run the setup script with different options:

```bash
# Full setup with all prompts
python3 setup_run_mcp.py

# Skip dependency installation
python3 setup_run_mcp.py --skip-deps

# Skip setup and run server 
python3 setup_run_mcp.py --run-server
```

## Start Server Directly

If you chose not to start the server during setup, you can start it manually:

```bash

# Using uv directly
source .venv/bin/activate
uv run python paprmcp.py

# For debugging run and use mcp inspector as client
source .venv/bin/activate
fastmcp dev paprmcp.py
```

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
### Debugging with VS Code

1. Install debugpy:
```bash
uv pip install ".[dev]" 
```

2. Start the server as well as mcp inspector in debug mode:
```bash
source .venv/bin/activate
python -m debugpy --wait-for-client --listen 5678 .venv/bin/fastmcp dev paprmcp.py
```

3. In VS Code:
   - Go to Run and Debug view (Ctrl+Shift+D or Cmd+Shift+D)
   - Select "Python: Attach to FastMCP"
   - Click the green play button or press F5
   - Set breakpoints in your code by clicking in the left margin
   - The debugger will stop at breakpoints when the code is executed


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

Remote MCP Todo
-> add support for http route in MCP server (p0)
-> host in the cloud (p0)


