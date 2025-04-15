from fastmcp import FastMCP
from typing import List, Dict, Optional, Callable, Any, Union
import httpx
import asyncio
import os
from dotenv import load_dotenv
import json
import functools
from models.memory_models import (
    ParseStoredMemory,
    NeoNode,
    RelatedMemoryResult,
    GetMemoryResponse,
    MemoryItem
)
from models.parse_server import (
    AddMemoryResponse,
    ParseUserPointer,
    ParsePointer,
    DeleteMemoryResponse,
    UpdateMemoryResponse,
    DeletionStatus,
    SystemUpdateStatus
)
from services.logging_config import get_logger

# Load environment variables
load_dotenv()

# Get logger instance
logger = get_logger(__name__)
logger.info("Logging system initialized")

def debug_tool(func: Callable) -> Callable:
    """Wrapper to enable breakpoints in mcp.tool functions"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Set breakpoint here to debug async functions
        logger.debug(f"Debug wrapper called for {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Debug wrapper result for {func.__name__}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Set breakpoint here to debug sync functions
        logger.debug(f"Debug wrapper called for {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Debug wrapper result for {func.__name__}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Create an MCP server
mcp = FastMCP("Demo 🚀")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    # Debug point - set breakpoint here
    _a = a  # Set breakpoint on this line
    _b = b
    logger.debug(f"Adding {_a} and {_b}")
    result = _a + _b
    logger.debug(f"Result: {result}")
    return result

@debug_tool
@mcp.tool()
async def get_memory(
    query: str    
) -> Dict:
    """
    Get memories from the Papr API based on a query.
    """
    api_url = os.getenv("MEMERY_SERVER_URL")
    if not api_url:
        raise ValueError("MEMERY_SERVER_URL is required for authentication")
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("PAPR_API_KEY is required for authentication")
    logger.debug(f"Getting memory for query: {query}")
    logger.debug(f"API URL: {api_url}")
    logger.debug(f"PAPR_API_KEY: {os.getenv('PAPR_API_KEY')}")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
        'Accept-Encoding': 'gzip'
    }

    data = {
        "query": query
    }

    try:
        # Create client with timeout configuration
        timeout = httpx.Timeout(30.0, connect=5.0)  # 30s total timeout, 5s connect timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/get_memory")
            response = await client.post(
                f"{api_url}/get_memory",
                json=data,
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")  # Added logging of raw response
            
            validated_response = GetMemoryResponse.model_validate(response_data)
            logger.debug(f"Validated response: {validated_response}")
            if not validated_response.success:
                raise ValueError(f"API returned unsuccessful response: {validated_response.error}")
            
            logger.debug(f"Successfully retrieved memories: {validated_response.model_dump()}")
            return validated_response.model_dump()

    except httpx.TimeoutException as e:
        logger.error(f"Request timed out: {str(e)}")
        raise ValueError(f"Request to {api_url} timed out after 30 seconds")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error getting memories: {str(e)}")
        raise

@debug_tool
@mcp.tool()
async def add_memory(
    content: str,
    type: str = "text",
    metadata: Optional[Dict] = None,
    context: Optional[Dict] = None,
    api_url: str = "https://memory.papr.ai",
) -> Dict:
    """
    Add a new memory to the memory store via the Papr API.
    """
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("PAPR_API_KEY is required for authentication")

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
        'Accept-Encoding': 'gzip'
    }

    data = {
        "content": content,
        "type": type,
        "metadata": metadata or {},
    }

    if context:
        data.update({"context": context})

    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Sending request to {api_url}/add_memory")
            response = await client.post(
                f"{api_url}/add_memory",
                json=data,
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            validated_response = AddMemoryResponse.model_validate(response_data)
            
            if validated_response.code != 200 or validated_response.status != "success":
                raise ValueError(f"API returned unsuccessful response: {response_data}")
            
            logger.debug(f"Successfully added memory: {validated_response.data[0].memoryId}")
            return validated_response.model_dump()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error adding memory: {str(e)}")
        raise

# Add a dynamic greeting resource
@debug_tool
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.tool()
async def delete_memory(memory_id: str, skip_parse: bool = False) -> DeleteMemoryResponse:
    """Delete a memory item by ID.
    
    Args:
        memory_id (str): ID of the memory to delete
        skip_parse (bool, optional): Skip Parse Server deletion. Defaults to False.
        
    Returns:
        DeleteMemoryResponse: Response containing deletion status
    """
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{mcp.base_url}/delete_memory",
            headers=mcp.headers,
            params={"id": memory_id, "skip_parse": skip_parse}
        )
        response.raise_for_status()
        return DeleteMemoryResponse.model_validate(response.json())

@mcp.tool()
async def update_memory(
    memory_id: str,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    type: Optional[str] = None
) -> UpdateMemoryResponse:
    """Update a memory item by ID.
    
    Args:
        memory_id (str): ID of the memory to update
        content (Optional[str]): New content for the memory
        metadata (Optional[Dict[str, Any]]): Updated metadata
        type (Optional[str]): Type of memory
        
    Returns:
        UpdateMemoryResponse: Response containing updated memory items
    """
    update_data = {}
    if content is not None:
        update_data["content"] = content
    if metadata is not None:
        update_data["metadata"] = metadata
    if type is not None:
        update_data["type"] = type

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{mcp.base_url}/update_memory",
            headers=mcp.headers,
            params={"id": memory_id},
            json=update_data
        )
        response.raise_for_status()
        return UpdateMemoryResponse.model_validate(response.json())

if __name__ == "__main__":
    try:
        # Run the MCP server
        logger.info("Starting MCP server...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}")
        raise

