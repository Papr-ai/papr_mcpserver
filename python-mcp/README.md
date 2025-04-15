# Papr Memory MCP

A FastMCP server implementation for interacting with the Papr Memory API. This project provides a simple interface for adding and managing memories using the Papr Memory service.

## Features

- Add memories with content, metadata, and context
- FastMCP integration for easy tool usage
- Async API support using httpx
- Environment-based configuration
- Type hints and Pydantic models for data validation
- VS Code debugging support

## Installation

1. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Configure environment:
- Copy `.env.example` to `.env`
- Add your Papr API key:
```
PAPR_API_KEY=your_api_key_here
```

## Usage

### Running the Server

Run the FastMCP server:
```bash
fastmcp dev paprmcp.py
```

### Debugging with VS Code

1. Install debugpy:
```bash
uv pip install debugpy
```

2. Start the server in debug mode:
```bash
python -m debugpy --wait-for-client --listen 5678 .venv/bin/fastmcp dev paprmcp.py
```

3. In VS Code:
   - Go to Run and Debug view (Ctrl+Shift+D or Cmd+Shift+D)
   - Select "Python: Attach to FastMCP"
   - Click the green play button or press F5
   - Set breakpoints in your code by clicking in the left margin
   - The debugger will stop at breakpoints when the code is executed

The server provides the following tools:
- `add_memory`: Add a new memory to the Papr Memory service
- `add`: Simple addition tool (demo)

## Development

This project uses:
- FastMCP for tool management
- httpx for async HTTP requests
- Pydantic for data validation
- python-dotenv for environment management

### Stopping the Server
```bash
pkill -f fastmcp
```

## License

MIT License 