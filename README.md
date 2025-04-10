# Papr Memory MCP

A FastMCP server implementation for interacting with the Papr Memory API. This project provides a simple interface for adding and managing memories using the Papr Memory service.

## Features

- Add memories with content, metadata, and context
- FastMCP integration for easy tool usage
- Async API support using httpx
- Environment-based configuration
- Type hints and Pydantic models for data validation

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

Run the FastMCP server:
```bash
fastmcp dev paprmcp.py
```

The server provides the following tools:
- `add_memory`: Add a new memory to the Papr Memory service
- `add`: Simple addition tool (demo)

## Development

This project uses:
- FastMCP for tool management
- httpx for async HTTP requests
- Pydantic for data validation
- python-dotenv for environment management

## License

MIT License 