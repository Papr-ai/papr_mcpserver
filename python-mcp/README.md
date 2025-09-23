# Papr MCP Server

A FastAPI-based MCP (Memory Control Protocol) server implementation for integrating with Papr's memory services (https://papr.ai).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../LICENSE) file for details.

## Prerequisites

- Python 3.10 or higher
- **Get your API key:** You can find it in the settings section of **[papr.ai](https://papr.ai)**. You'll need to create a developer account on Papr to get your API key.

## Quick Start (Recommended)

Run the universal setup script and follow the prompts:

```bash
python setup_mcp.py
```

What this does:
- Lets you choose your application (Cursor, Claude, or Other)
- Uses the PyPI package `papr-memory-mcp` with `uv` for automatic installation
- Prompts for your Papr API key
- Creates/updates the MCP configuration in the correct location
- For "Other", prints the JSON you can copy into your client's config

After setup, restart your selected application.

> **Note:** This method uses the published PyPI package `papr-memory-mcp` which is automatically installed and managed by `uv`. No local code changes needed.

## Alternative: Run Locally (Developer Mode)

If you prefer to run the MCP server locally from source:

1) Clone and enter the repo
```bash
git clone https://github.com/Papr-ai/papr_mcpserver
cd papr_mcpserver/python-mcp
```

2) Create and activate a virtual environment (recommended)
```bash
uv venv
source .venv/bin/activate    # macOS/Linux
# or on Windows PowerShell
.venv\Scripts\Activate.ps1
```

3) Install dependencies
```bash
uv pip install -e .[dev]
```

4) Set your API key
```bash
export PAPR_API_KEY=your_api_key_here        # macOS/Linux
setx PAPR_API_KEY your_api_key_here          # Windows (new shells)
$env:PAPR_API_KEY="your_api_key_here"       # Windows PowerShell (current shell)
```

5) Start the server
```bash
uv run python paprmcp.py
# or for interactive development
fastmcp dev paprmcp.py
```

6) Point your client to the server
- Use the JSON from the README (or run `python setup_mcp.py` and pick "Other") to configure your client’s `mcp.json`.

## Start Server Directly

If you chose not to start the server during setup, you can start it manually:

**On macOS/Linux:**
```bash
# Using uv directly
source .venv/bin/activate
uv run python paprmcp.py

# For debugging run and use mcp inspector as client
source .venv/bin/activate
fastmcp dev paprmcp.py
```

**On Windows:**
```cmd
# Using uv directly
.venv\Scripts\activate
uv run python paprmcp.py

# For debugging run and use mcp inspector as client
.venv\Scripts\activate
fastmcp dev paprmcp.py
```

**On Windows PowerShell:**
```powershell
# Using uv directly
.venv\Scripts\Activate.ps1
uv run python paprmcp.py

# For debugging run and use mcp inspector as client
.venv\Scripts\Activate.ps1
fastmcp dev paprmcp.py
```

Note: Using the setup script with `--run-server` is recommended as it ensures the correct virtual environment is used and proper configuration is loaded.

## Created Configuration

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

2. **For MCP Inspector (optional):** Install Node.js to get npx:
```bash
# On Windows (using winget)
winget install OpenJS.NodeJS
# After installation, refresh PATH in PowerShell:
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# Also ensure uv is properly installed and in PATH:
$env:PATH = "C:\Users\$env:USERNAME\.local\bin;$env:PATH"

# On macOS (using Homebrew)
brew install node

# On Linux (using package manager)
# Ubuntu/Debian:
sudo apt update && sudo apt install nodejs npm
# Or using NodeSource repository for latest version:
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```

3. Start the server as well as mcp inspector in debug mode:

**On macOS/Linux:**
```bash
source .venv/bin/activate
python -m debugpy --wait-for-client --listen 5678 .venv/bin/fastmcp dev paprmcp.py
```

**On Windows:**
```cmd
.venv\Scripts\activate
python -m debugpy --wait-for-client --listen 5678 .venv\Scripts\fastmcp.exe dev paprmcp.py
```

**On Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
python -m debugpy --wait-for-client --listen 5678 .venv\Scripts\fastmcp.exe dev paprmcp.py
```

4. In VS Code:
   - Go to Run and Debug view (Ctrl+Shift+D or Cmd+Shift+D)
   - Select "Python: Attach to FastMCP"
   - Click the green play button or press F5
   - Set breakpoints in your code by clicking in the left margin
   - The debugger will stop at breakpoints when the code is executed

5. **Using MCP Inspector (alternative to VS Code debugging):**
   - After starting the server with `fastmcp dev paprmcp.py`, you can use the MCP inspector
   - The inspector will automatically connect to your running MCP server
   - This provides a web-based interface to test and interact with your MCP tools


## Troubleshooting

If you encounter any issues:

1. **Python command not found:**
   - If `python3` is not found, try using `python` instead
   - Check your Python installation: `python --version` or `python3 --version`
   - On Windows, Python 3 is often installed as `python` rather than `python3`

2. **Windows-specific issues:**
   - **PowerShell execution policy:** If you get execution policy errors, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
   - **Virtual environment activation:** Use `.venv\Scripts\activate` in Command Prompt or `.venv\Scripts\Activate.ps1` in PowerShell
   - **Path issues:** Ensure `uv` is in your PATH. It's typically installed to `%USERPROFILE%\.cargo\bin` or `%LOCALAPPDATA%\uv\bin`

3. **General issues:**
   - Check the logs for detailed error messages
   - Ensure your Papr API key is correctly set in the `.env` file
   - Verify the virtual environment is activated
   - Make sure all dependencies are installed correctly

For additional help, please contact support or open an issue in the repository.


