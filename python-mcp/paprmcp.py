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
mcp = FastMCP("Papr MCP Server")


@mcp.tool()
@debug_tool
async def get_memory(
    query: str,
    project_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    relation_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_memory_items: Optional[int] = None
) -> Dict:
    """
    Get memories from the Papr API based on a query.
    
    Args:
        query (str): The search query string
        project_id (Optional[str]): Project ID to filter memories
        context (Optional[Dict]): Additional context for the search
        relation_type (Optional[str]): Type of relation to search for
        metadata (Optional[Dict]): Additional metadata for filtering
        max_memory_items (Optional[int]): Maximum number of memory items to return
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

    # Add optional fields if provided
    if project_id is not None:
        data["project_id"] = project_id
    if context is not None:
        data["context"] = context
    if relation_type is not None:
        data["relation_type"] = relation_type
    if metadata is not None:
        data["metadata"] = metadata

    # Prepare query parameters
    params = {}
    if max_memory_items is not None:
        params["max_memory_items"] = max_memory_items

    try:
        # Create client with timeout configuration
        timeout = httpx.Timeout(30.0, connect=5.0)  # 30s total timeout, 5s connect timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/get_memory")
            logger.debug(f"Request data: {data}")
            logger.debug(f"Query params: {params}")
            response = await client.post(
                f"{api_url}/get_memory",
                json=data,
                params=params,
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
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

@mcp.tool()
@debug_tool
async def add_memory(
    content: str,
    type: str = "text",
    metadata: Optional[Dict] = None,
    context: Optional[Dict] = None,
    skip_background_processing: bool = False
) -> Dict:
    """
    Add a new memory to the memory store via the Papr API.
    
    Args:
        content (str): The content of the memory
        type (str): Type of memory (default: "text")
        metadata (Optional[Dict]): Additional metadata for the memory
        context (Optional[Dict]): Context information for the memory
        skip_background_processing (bool): If True, skips adding background tasks for processing
    """
    api_url = os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai")
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
        # Create client with timeout configuration
        timeout = httpx.Timeout(30.0, connect=5.0)  # 30s total timeout, 5s connect timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/add_memory")
            logger.debug(f"Request data: {data}")
            logger.debug(f"Skip background processing: {skip_background_processing}")
            
            response = await client.post(
                f"{api_url}/add_memory",
                json=data,
                params={"skip_background_processing": skip_background_processing},
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
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

@mcp.tool()
@debug_tool
async def add_document(
    content: str,
    type: str = "document",
    metadata: Optional[Dict] = None,
    context: Optional[Dict] = None,
    skip_background_processing: bool = False
) -> Dict:
    """
    Add a document to the memory store via the Papr API.
    
    Args:
        content (str): The content of the document
        type (str): Type of document (default: "document")
        metadata (Optional[Dict]): Additional metadata for the document
        context (Optional[Dict]): Context information for the document
        skip_background_processing (bool): If True, skips adding background tasks for processing
    """
    api_url = os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai")
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
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/add_memory")
            logger.debug(f"Request data: {data}")
            logger.debug(f"Skip background processing: {skip_background_processing}")
            
            response = await client.post(
                f"{api_url}/add_memory",
                json=data,
                params={"skip_background_processing": skip_background_processing},
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
            validated_response = AddMemoryResponse.model_validate(response_data)
            
            if validated_response.code != 200 or validated_response.status != "success":
                raise ValueError(f"API returned unsuccessful response: {response_data}")
            
            logger.debug(f"Successfully added document: {validated_response.data[0].memoryId}")
            return validated_response.model_dump()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise

@mcp.tool()
@debug_tool
async def add_memory_batch(
    memories: List[Dict[str, Any]],
    batch_size: int = 5,
    skip_background_processing: bool = False
) -> Dict:
    """
    Add multiple memory items in a batch.
    
    Args:
        memories (List[Dict]): List of memory items to add
        batch_size (int): Number of items to process in each batch (default: 5)
        skip_background_processing (bool): If True, skips adding background tasks for processing
    """
    api_url = os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai")
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("PAPR_API_KEY is required for authentication")

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
        'Accept-Encoding': 'gzip'
    }

    try:
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/add_memory_batch")
            logger.debug(f"Batch size: {batch_size}")
            logger.debug(f"Skip background processing: {skip_background_processing}")
            
            response = await client.post(
                f"{api_url}/add_memory_batch",
                json={"memories": memories},
                params={
                    "batch_size": batch_size,
                    "skip_background_processing": skip_background_processing
                },
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
            validated_response = BatchMemoryResponse.model_validate(response_data)
            return validated_response.model_dump()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error adding memory batch: {str(e)}")
        raise

@mcp.tool()
@debug_tool
async def add_memory_batch_with_details(
    memories: List[Dict[str, Any]],
    tenant_subtenant: str,
    connector: str,
    stream: str,
    batch_size: int = 5,
    skip_background_processing: bool = False
) -> Dict:
    """
    Add multiple memory items in a batch with tenant/connector/stream details.
    
    Args:
        memories (List[Dict]): List of memory items to add
        tenant_subtenant (str): Tenant and subtenant identifier
        connector (str): Connector identifier
        stream (str): Stream identifier
        batch_size (int): Number of items to process in each batch (default: 5)
        skip_background_processing (bool): If True, skips adding background tasks for processing
    """
    api_url = os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai")
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("PAPR_API_KEY is required for authentication")

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
        'Accept-Encoding': 'gzip'
    }

    try:
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/add_memory_batch/{tenant_subtenant}/{connector}/{stream}")
            logger.debug(f"Batch size: {batch_size}")
            logger.debug(f"Skip background processing: {skip_background_processing}")
            
            response = await client.post(
                f"{api_url}/add_memory_batch/{tenant_subtenant}/{connector}/{stream}",
                json={"memories": memories},
                params={
                    "batch_size": batch_size,
                    "skip_background_processing": skip_background_processing
                },
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
            validated_response = BatchMemoryResponse.model_validate(response_data)
            return validated_response.model_dump()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error adding memory batch with details: {str(e)}")
        raise

# Add a dynamic greeting resource
@mcp.resource("greeting://{Papr}")
@debug_tool
def get_greeting(Papr: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {Papr}!"

@mcp.tool()
@debug_tool
async def delete_memory(
    memory_id: str,
    skip_parse: bool = False
) -> Dict:
    """
    Delete a memory item by ID.
    
    Args:
        memory_id (str): ID of the memory to delete
        skip_parse (bool, optional): Skip Parse Server deletion. Defaults to False.
        
    Returns:
        DeleteMemoryResponse: Response containing deletion status
    """
    api_url = os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai")
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("PAPR_API_KEY is required for authentication")

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
        'Accept-Encoding': 'gzip'
    }

    try:
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/delete_memory")
            logger.debug(f"Memory ID: {memory_id}")
            logger.debug(f"Skip Parse: {skip_parse}")
            
            response = await client.delete(
                f"{api_url}/delete_memory",
                params={
                    "id": memory_id,
                    "skip_parse": skip_parse
                },
                headers=headers
            )
            
            # Handle both 200 (success) and 207 (partial success) responses
            if response.status_code not in [200, 207]:
                response.raise_for_status()
                
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
            validated_response = DeleteMemoryResponse.model_validate(response_data)
            return validated_response.model_dump()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise

@mcp.tool()
@debug_tool
async def update_memory(
    memory_id: str,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    type: Optional[str] = None
) -> Dict:
    """
    Update a memory item by ID.
    
    Args:
        memory_id (str): ID of the memory to update
        content (Optional[str]): New content for the memory
        metadata (Optional[Dict[str, Any]]): Updated metadata
        type (Optional[str]): Type of memory
        
    Returns:
        UpdateMemoryResponse: Response containing updated memory items
    """
    api_url = os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai")
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("PAPR_API_KEY is required for authentication")

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
        'Accept-Encoding': 'gzip'
    }

    # Prepare update data
    update_data = {}
    if content is not None:
        update_data["content"] = content
    if metadata is not None:
        update_data["metadata"] = metadata
    if type is not None:
        update_data["type"] = type

    try:
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/update_memory")
            logger.debug(f"Memory ID: {memory_id}")
            logger.debug(f"Update data: {update_data}")
            
            response = await client.put(
                f"{api_url}/update_memory",
                params={"id": memory_id},
                json=update_data,
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
            validated_response = UpdateMemoryResponse.model_validate(response_data)
            return validated_response.model_dump()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error updating memory: {str(e)}")
        raise

@mcp.tool()
@debug_tool
async def get_document_status(
    memory_id: str
) -> Dict:
    """
    Get the status of a document in the memory store.
    
    Args:
        memory_id (str): ID of the document to check status for
        
    Returns:
        Dict: Response containing document status information including:
            - processing_status: Current processing state
            - chunk_count: Number of chunks processed
            - error: Any error message if processing failed
            - last_updated: Timestamp of last status update
    """
    api_url = os.getenv("MEMERY_SERVER_URL", "https://memory.papr.ai")
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("PAPR_API_KEY is required for authentication")

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'Authorization': f'APIKey {os.getenv("PAPR_API_KEY")}',
        'Accept-Encoding': 'gzip'
    }

    try:
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to {api_url}/document_status")
            logger.debug(f"Memory ID: {memory_id}")
            
            response = await client.get(
                f"{api_url}/document_status",
                params={"id": memory_id},
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Raw response: {response_data}")
            
            # Validate response structure
            if not isinstance(response_data, dict):
                raise ValueError("Invalid response format")
            
            required_fields = ["processing_status", "chunk_count", "last_updated"]
            for field in required_fields:
                if field not in response_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return response_data

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise

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

