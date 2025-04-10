from fastmcp import FastMCP
from typing import List, Dict, Optional
import httpx
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryItem(BaseModel):
    memoryId: str
    objectId: str
    createdAt: datetime
    memoryChunkIds: List[str]

class AddMemoryResponse(BaseModel):
    code: int
    status: str
    data: List[MemoryItem]

# Create an MCP server
mcp = FastMCP("Demo 🚀")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

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
    
    Args:
        content: The content of the memory to store
        type: Type of memory (default: 'text')
        metadata: Optional metadata for the memory
        context: Optional context information
        api_url: The base URL for the Papr API
        
    
    Returns:
        dict: The validated response from the memory API
    """
    if not os.getenv("PAPR_API_KEY"):
        raise ValueError("session_token is required for authentication")

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
            logger.info(f"Sending request to {api_url}/add_memory")
            response = await client.post(
                f"{api_url}/add_memory",
                json=data,
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Validate response using Pydantic model
            validated_response = AddMemoryResponse.model_validate(response_data)
            
            if validated_response.code != 200 or validated_response.status != "success":
                raise ValueError(f"API returned unsuccessful response: {response_data}")
            
            logger.info(f"Successfully added memory: {validated_response.data[0].memoryId}")
            return validated_response.model_dump()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error adding memory: {str(e)}")
        raise

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"
