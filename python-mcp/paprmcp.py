from fastmcp import FastMCP
from fastmcp.server.openapi import FastMCPOpenAPI, RouteMap, RouteType
from fastapi import FastAPI
from typing import List, Dict, Optional, Callable, Any, Union
from pydantic import BaseModel
import httpx
import asyncio
import os
from dotenv import load_dotenv
import json
import functools
import logging
from services.logging_config import get_logger
from mcp.types import TextContent, ImageContent, EmbeddedResource
from typing import Any, List
import json
import logging
from pathlib import Path
import yaml
import sys
import traceback
import tempfile

# Add immediate stderr output for debugging
print("=== PAPR MCP SERVER STARTING ===", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Working directory: {os.getcwd()}", file=sys.stderr)

# Load environment variables
load_dotenv()
print("Environment variables loaded", file=sys.stderr)

# Get logger instance
logger = get_logger(__name__)
logger.info("Logging system initialized")
print("Logger initialized", file=sys.stderr)

# Setup basic configuration
api_key = os.getenv("PAPR_API_KEY")
if api_key:
    logger.info(f"API key loaded: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    print(f"API key loaded: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}", file=sys.stderr)
else:
    logger.warning("No API key found in environment variables!")
    print("WARNING: No API key found in environment variables!", file=sys.stderr)

server_url = os.getenv("MEMORY_SERVER_URL", "https://memory.papr.ai")
logger.info(f"Connecting to server: {server_url}")
print(f"Connecting to server: {server_url}", file=sys.stderr)

class CustomFastMCP(FastMCPOpenAPI):
    def __init__(
        self,
        openapi_spec: dict[str, Any],
        client: httpx.AsyncClient,
        name: str | None = None,
        route_maps: list[RouteMap] | None = None,
        **settings: Any,
    ):
        print("Initializing CustomFastMCP...", file=sys.stderr)
        super().__init__(
            openapi_spec=openapi_spec,
            client=client,
            name=name or "Papr Memory MCP",
            route_maps=route_maps,
            **settings
        )
        logger.info("CustomFastMCP initialized with OpenAPI spec")
        print("CustomFastMCP initialized with OpenAPI spec", file=sys.stderr)
        logger.info(f"Registered tools: {list(self._tool_manager._tools.keys())}")
        print(f"Registered tools: {list(self._tool_manager._tools.keys())}", file=sys.stderr)
        
        # Override the tool manager's call_tool method
        original_call_tool = self._tool_manager.call_tool
        
        async def custom_call_tool(name: str, arguments: dict[str, Any], context: Any = None) -> Any:
            logger.info(f"Custom call_tool called with name={name}, arguments={arguments}")
            print(f"Custom call_tool called with name={name}, arguments={arguments}", file=sys.stderr)
            try:
                result = await original_call_tool(name, arguments, context)
                logger.info(f"Custom call_tool result: {result}")
                print(f"Custom call_tool result: {result}", file=sys.stderr)
                
                # If the result is a dictionary
                if isinstance(result, dict):
                    # Check if it's an API response with 'data' field
                    if 'data' in result:
                        return [TextContent(text=json.dumps(result['data']), type="text")]
                    # Check if it's an error response
                    elif 'error' in result or 'detail' in result:
                        return [TextContent(text=json.dumps(result), type="text")]
                    # For other dictionary responses
                    return [TextContent(text=json.dumps(result), type="text")]
                
                # If the result is already a list of content objects
                if isinstance(result, list) and all(
                    isinstance(item, (TextContent, ImageContent, EmbeddedResource))
                    for item in result
                ):
                    return result
                
                # For string results
                if isinstance(result, str):
                    return [TextContent(text=result, type="text")]
                
                # For any other type
                return [TextContent(text=str(result), type="text")]
            except Exception as e:
                logger.error(f"Error in custom_call_tool: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                print(f"ERROR in custom_call_tool: {str(e)}", file=sys.stderr)
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                raise
        
        # Replace the tool manager's call_tool method
        self._tool_manager.call_tool = custom_call_tool

def init_mcp():
    """Initialize MCP server with OpenAPI spec and HTTP client"""
    try:
        print("Initializing MCP server...", file=sys.stderr)
        
        # Setup HTTP client and headers
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': api_key if api_key else '',
            'Accept-Encoding': 'gzip'
        }
        logger.info(f"Headers: {headers}")
        print(f"Headers configured: {headers}", file=sys.stderr)

        # Get proxy settings from environment
        http_proxy = os.getenv("HTTP_PROXY")
        https_proxy = os.getenv("HTTPS_PROXY")
        
        print(f"Creating HTTP client...", file=sys.stderr)
        http_client = httpx.AsyncClient(
            base_url=server_url,
            headers=headers,
            proxy=http_proxy or https_proxy
        )
        logger.info(f"HTTP client created: {http_client}")
        print(f"HTTP client created successfully", file=sys.stderr)

        # Fetch OpenAPI YAML from server and convert to JSON
        def get_openapi_schema_sync():
            """Synchronous version of get_openapi_schema"""
            try:
                print("Fetching OpenAPI schema...", file=sys.stderr)
                import requests
                response = requests.get(f"{server_url}/openapi.yaml")
                print(f"OpenAPI response status: {response.status_code}", file=sys.stderr)
                if response.status_code == 200:
                    # Convert YAML to JSON
                    yaml_content = response.text
                    print(f"OpenAPI YAML content length: {len(yaml_content)}", file=sys.stderr)
                    return yaml.safe_load(yaml_content)
                else:
                    logger.error(f"Failed to fetch OpenAPI YAML: {response.status_code}")
                    print(f"ERROR: Failed to fetch OpenAPI YAML: {response.status_code}", file=sys.stderr)
                    raise Exception(f"Failed to fetch OpenAPI YAML: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching OpenAPI YAML: {str(e)}")
                print(f"ERROR: Error fetching OpenAPI YAML: {str(e)}", file=sys.stderr)
                raise

        # Get OpenAPI schema synchronously
        print("Getting OpenAPI schema...", file=sys.stderr)
        openapi_spec = get_openapi_schema_sync()
        print("OpenAPI schema fetched successfully", file=sys.stderr)
        
        # Dump OpenAPI spec to a writable location for debugging/reference
        try:
            # Try to write to logs directory first
            logs_dir = Path("logs")
            if logs_dir.exists() and os.access(logs_dir, os.W_OK):
                spec_path = logs_dir / "openapi_spec.json"
            else:
                # Fall back to temp directory
                spec_path = Path(tempfile.gettempdir()) / "openapi_spec.json"
            
            with open(spec_path, "w") as f:
                json.dump(openapi_spec, f, indent=2)
            logger.info(f"Dumped OpenAPI spec to {spec_path}")
            print(f"Dumped OpenAPI spec to {spec_path}", file=sys.stderr)
        except Exception as e:
            logger.warning(f"Could not dump OpenAPI spec to file: {e}")
            print(f"Warning: Could not dump OpenAPI spec to file: {e}", file=sys.stderr)
            # Continue without dumping the file
        
        # Create MCP instance with OpenAPI spec using CustomFastMCP
        mcp = CustomFastMCP(
            openapi_spec=openapi_spec,
            client=http_client,
            name="Papr Memory MCP"
        )
        
        # Log the tools that were registered
        logger.info(f"Initialized MCP with tools: {list(mcp._tool_manager._tools.keys())}")
        print(f"Initialized MCP with tools: {list(mcp._tool_manager._tools.keys())}", file=sys.stderr)
        return mcp
    except Exception as e:
        logger.error(f"Error initializing MCP: {str(e)}")
        print(f"ERROR initializing MCP: {str(e)}", file=sys.stderr)
        raise

# Try to initialize the full MCP with OpenAPI spec
print("Attempting to initialize MCP with OpenAPI spec...", file=sys.stderr)
try:
    mcp = init_mcp()
    print("Successfully initialized MCP with OpenAPI spec", file=sys.stderr)
    print(f"Available tools: {list(mcp._tool_manager._tools.keys())}", file=sys.stderr)
except Exception as e:
    print(f"Failed to initialize MCP with OpenAPI spec: {e}", file=sys.stderr)
    print("Falling back to basic MCP...", file=sys.stderr)
    
    # Fallback to basic MCP if OpenAPI initialization fails
    mcp = FastMCP("Papr Memory MCP")
    
    @mcp.tool()
    async def get_memories(query: str = None) -> str:
        """Get memories from Papr Memory API"""
        return f"Memory search for: {query} (OpenAPI not available)"

    @mcp.tool()
    async def add_memory(content: str) -> str:
        """Add a memory to Papr Memory API"""
        return f"Memory added: {content} (OpenAPI not available)"

print("Module initialization completed successfully", file=sys.stderr)

if __name__ == "__main__":
    try:
        # Start the server
        print("=== STARTING MCP SERVER ===", file=sys.stderr)
        logger.info("Starting MCP server process...")
        logger.info("About to call mcp.run()...")
        print("About to call mcp.run()...", file=sys.stderr)
        
        # Use FastMCP's run method
        mcp.run()
        print("MCP server finished running", file=sys.stderr)
        logger.info("MCP server finished running")
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...", file=sys.stderr)
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"ERROR running MCP server: {str(e)}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        logger.error(f"Error running MCP server: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise



