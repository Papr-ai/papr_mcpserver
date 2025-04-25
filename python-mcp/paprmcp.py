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

# Load environment variables
load_dotenv()

# Get logger instance
logger = get_logger(__name__)
logger.info("Logging system initialized")

# Global variables

http_client = None

# Setup HTTP client and headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
    'Accept-Encoding': 'gzip'
}

http_client = httpx.AsyncClient(
    base_url=os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai"),
    headers=headers
)

# Get the directory containing the script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Load OpenAPI spec using absolute path
openapi_path = SCRIPT_DIR / "openapi.json"
try:
    with open(openapi_path, "r") as f:
        openapi_spec = json.load(f)
except FileNotFoundError:
    logger.error(f"OpenAPI spec not found at {openapi_path}")
    raise

logger = logging.getLogger(__name__)

class CustomFastMCP(FastMCPOpenAPI):
    def __init__(
        self,
        openapi_spec: dict[str, Any],
        client: httpx.AsyncClient,
        name: str | None = None,
        route_maps: list[RouteMap] | None = None,
        **settings: Any,
    ):
        super().__init__(
            openapi_spec=openapi_spec,
            client=client,
            name=name or "Papr Memory MCP",
            route_maps=route_maps,
            **settings
        )
        logger.info("CustomFastMCP initialized with OpenAPI spec")
        logger.info(f"Registered tools: {list(self._tool_manager._tools.keys())}")
        
        # Override the tool manager's call_tool method
        original_call_tool = self._tool_manager.call_tool
        
        async def custom_call_tool(name: str, arguments: dict[str, Any], context: Any = None) -> Any:
            logger.info(f"Custom call_tool called with name={name}, arguments={arguments}")
            result = await original_call_tool(name, arguments, context)
            logger.info(f"Custom call_tool result: {result}")
            
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
        
        # Replace the tool manager's call_tool method
        self._tool_manager.call_tool = custom_call_tool

  
                
def init_mcp():
    """Initialize MCP server with OpenAPI spec and HTTP client"""
    try:
        # Create MCP instance with OpenAPI spec using CustomFastMCP
        mcp = CustomFastMCP(
            openapi_spec=openapi_spec,
            client=http_client,
            name="Papr Memory MCP"
        )
        
        # Log the tools that were registered
        logger.info(f"Initialized MCP with tools: {list(mcp._tool_manager._tools.keys())}")
        return mcp
    except Exception as e:
        logger.error(f"Error initializing MCP: {str(e)}")
        raise

# Initialize Papr MCP
mcp = init_mcp()

if __name__ == "__main__":
    try:
        # Start the server
        logger.info("Starting MCP server process...")
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        if http_client:
            asyncio.run(http_client.aclose())
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}")
        if http_client:
            asyncio.run(http_client.aclose())
        raise



