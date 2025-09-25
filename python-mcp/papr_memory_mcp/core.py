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
from mcp.types import TextContent, ImageContent, EmbeddedResource
from typing import Any, List
import json
import logging
from pathlib import Path
import yaml
import sys
import traceback
import tempfile

# Import logger with fallback for different contexts
try:
    from .services.logging_config import get_logger
except ImportError:
    try:
        from services.logging_config import get_logger
    except ImportError:
        # Fallback to basic logging if services module not available
        def get_logger(name):
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger

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

        # The JSON Schema endpoint already provides the correct format with $defs
        # No conversion needed since we're using /openapi-json-schema.json

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
        
        async def custom_call_tool(*args, **kwargs) -> Any:
            logger.info(f"Custom call_tool called with args={args}, kwargs={kwargs}")
            print(f"Custom call_tool called with args={args}, kwargs={kwargs}", file=sys.stderr)
            try:
                # Extract name and arguments from the call
                if len(args) >= 2:
                    name = args[0]
                    arguments = args[1]
                elif 'name' in kwargs and 'arguments' in kwargs:
                    name = kwargs['name']
                    arguments = kwargs['arguments']
                elif 'key' in kwargs and 'arguments' in kwargs:
                    # MCP inspector uses 'key' instead of 'name'
                    name = kwargs['key']
                    arguments = kwargs['arguments']
                else:
                    raise ValueError(f"Could not extract name and arguments from call. Args: {args}, Kwargs: {kwargs}")
                
                logger.info(f"Extracted name={name}, arguments={arguments}")
                print(f"Extracted name={name}, arguments={arguments}", file=sys.stderr)
                
                # Call the original function with the correct arguments
                # Try different argument combinations to handle various calling patterns
                try:
                    if len(args) >= 3:
                        # Called with 3+ positional arguments
                        result = await original_call_tool(*args)
                    elif 'context' in kwargs:
                        result = await original_call_tool(name, arguments, kwargs['context'])
                    else:
                        result = await original_call_tool(name, arguments)
                except TypeError as e:
                    # If the above fails, try with just name and arguments
                    logger.warning(f"First call attempt failed: {e}, trying alternative")
                    try:
                        result = await original_call_tool(name, arguments)
                    except TypeError as e2:
                        # If that fails too, try with all kwargs
                        logger.warning(f"Second call attempt failed: {e2}, trying with kwargs")
                        result = await original_call_tool(name, arguments, **kwargs)
                logger.info(f"Custom call_tool result: {result}")
                print(f"Custom call_tool result: {result}", file=sys.stderr)
                
                # Return the result as-is - let FastMCP handle the conversion
                # The original call_tool method already returns the correct format
                return result
            except Exception as e:
                logger.error(f"Error in custom_call_tool: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                print(f"ERROR in custom_call_tool: {str(e)}", file=sys.stderr)
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                raise
        
        # Replace the tool manager's call_tool method
        self._tool_manager.call_tool = custom_call_tool
        
        # Filter tools to show only memory-related tools
        self._filter_memory_tools()
    
    @staticmethod
    def _convert_openapi_to_json_schema(openapi_spec: dict[str, Any]) -> dict[str, Any]:
        """
        Convert OpenAPI 3.1 spec to JSON Schema format for MCP compatibility.
        This properly converts components/schemas to $defs and updates all references.
        """
        import json
        import re
        
        converted_spec = openapi_spec.copy()
        
        # Add $defs alongside components.schemas for MCP compatibility
        if "components" in converted_spec and "schemas" in converted_spec["components"]:
            # Create a deep copy of the schemas to avoid reference issues
            import copy
            converted_spec["$defs"] = copy.deepcopy(converted_spec["components"]["schemas"])
            print("Added $defs alongside components.schemas for MCP compatibility", file=sys.stderr)
            print(f"DEBUG: $defs created with {len(converted_spec['$defs'])} schemas", file=sys.stderr)
            print(f"DEBUG: Memory in $defs: {'Memory' in converted_spec['$defs']}", file=sys.stderr)
            
            # Convert all references from #/components/schemas/ to #/$defs/
            def convert_refs(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == "$ref" and isinstance(value, str):
                            # Convert OpenAPI references to JSON Schema references
                            if value.startswith("#/components/schemas/"):
                                obj[key] = value.replace("#/components/schemas/", "#/$defs/")
                            elif value.startswith("#/components/"):
                                obj[key] = value.replace("#/components/", "#/$defs/")
                        else:
                            convert_refs(value)
                elif isinstance(obj, list):
                    for item in obj:
                        convert_refs(item)
            
            # Convert all references in the spec
            convert_refs(converted_spec)
            print("Converted all schema references from OpenAPI to JSON Schema format", file=sys.stderr)
            
            # Ensure $defs is at the root level and properly formatted
            if "$defs" not in converted_spec:
                print("ERROR: $defs not found after conversion!", file=sys.stderr)
            else:
                print(f"SUCCESS: $defs section created with {len(converted_spec['$defs'])} schemas", file=sys.stderr)
                if "Memory" in converted_spec["$defs"]:
                    print("SUCCESS: Memory schema found in $defs", file=sys.stderr)
                else:
                    print("ERROR: Memory schema not found in $defs", file=sys.stderr)
                    print(f"Available schemas: {list(converted_spec['$defs'].keys())[:10]}", file=sys.stderr)
        
        return converted_spec
    
    def _filter_memory_tools(self):
        """Filter tools to show only memory-related tools"""
        # Define the allowed memory tools
        allowed_tools = {
            'add_memory', 'get_memory', 'update_memory', 'delete_memory', 'search_memory', 'submit_feedback','submit_batch_feedback','add_memory_batch'
        }
        
        # Get all current tools
        all_tools = dict(self._tool_manager._tools)
        
        # Filter to only memory tools
        filtered_tools = {name: tool for name, tool in all_tools.items() if name in allowed_tools}
        
        # Replace the tools dictionary
        self._tool_manager._tools = filtered_tools
        
        logger.info(f"Filtered tools to memory-only: {list(filtered_tools.keys())}")
        print(f"Filtered tools to memory-only: {list(filtered_tools.keys())}", file=sys.stderr)

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
        def get_openapi_json_sync():
            """Synchronous version of get_openapi_json"""
            try:
                print("Fetching OpenAPI schema...", file=sys.stderr)
                import requests
                import time
                # Add cache-busting parameters to bypass caching
                cache_buster = int(time.time() * 1000)  # milliseconds timestamp
                url = f"{server_url}/openapi.json?t={cache_buster}&_cb={cache_buster}"
                headers = {
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
                response = requests.get(url, headers=headers)
                print(f"OpenAPI JSON response status: {response.status_code}", file=sys.stderr)
                if response.status_code == 200:
                    # Parse JSON directly
                    json_content = response.text
                    print(f"OpenAPI JSON content length: {len(json_content)}", file=sys.stderr)
                    return json.loads(json_content)
                else:
                    logger.error(f"Failed to fetch OpenAPI JSON: {response.status_code}")
                    print(f"ERROR: Failed to fetch OpenAPI JSON: {response.status_code}", file=sys.stderr)
                    raise Exception(f"Failed to fetch OpenAPI JSON: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching OpenAPI JSON: {str(e)}")
                print(f"ERROR: Error fetching OpenAPI JSON: {str(e)}", file=sys.stderr)
                raise

        # Get OpenAPI JSON synchronously
        print("Getting OpenAPI JSON...", file=sys.stderr)
        openapi_spec = get_openapi_json_sync()
        print("OpenAPI JSON fetched successfully", file=sys.stderr)
        
        # Dump OpenAPI JSON to a writable location for debugging/reference
        try:
            # Try to write to logs directory first
            logs_dir = Path("logs")
            if logs_dir.exists() and os.access(logs_dir, os.W_OK):
                spec_path = logs_dir / "openapi.json"
            else:
                # Fall back to temp directory
                spec_path = Path(tempfile.gettempdir()) / "openapi.json"
            
            with open(spec_path, "w") as f:
                json.dump(openapi_spec, f, indent=2)
            logger.info(f"Dumped OpenAPI JSON to {spec_path}")
            print(f"Dumped OpenAPI JSON to {spec_path}", file=sys.stderr)
        except Exception as e:
            logger.warning(f"Could not dump OpenAPI JSON to file: {e}")
            print(f"Warning: Could not dump OpenAPI JSON to file: {e}", file=sys.stderr)
            # Continue without dumping the file
        
        # Create MCP instance with OpenAPI JSON using CustomFastMCP
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

def main():
    """Main entry point for the Papr MCP server."""
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

if __name__ == "__main__":
    main()
