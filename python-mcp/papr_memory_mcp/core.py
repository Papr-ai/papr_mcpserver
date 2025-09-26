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
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
import logging
import traceback
import sys

# Import papr-memory SDK
try:
    from papr_memory import Papr
except ImportError:
    print("ERROR: papr-memory SDK not found. Please install it with: pip install papr-memory", file=sys.stderr)
    raise ImportError("papr-memory SDK is required but not installed")

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

# Initialize Papr Memory SDK client
try:
    papr_client = Papr(x_api_key=api_key)
    logger.info("Papr Memory SDK client initialized")
    print("Papr Memory SDK client initialized", file=sys.stderr)
except Exception as e:
    logger.error(f"Failed to initialize Papr Memory SDK: {str(e)}")
    print(f"ERROR: Failed to initialize Papr Memory SDK: {str(e)}", file=sys.stderr)
    raise

class CustomFastMCP(FastMCP):
    def __init__(self, name: str = "Papr Memory MCP", **settings):
        print("Initializing CustomFastMCP with explicit tools...", file=sys.stderr)
        super().__init__(name=name, **settings)
        
        # Register the 8 explicit memory tools
        self._register_memory_tools()
        
        logger.info("CustomFastMCP initialized with explicit memory tools")
        print("CustomFastMCP initialized with explicit memory tools", file=sys.stderr)
        logger.info(f"Registered tools: {list(self._tool_manager._tools.keys())}")
        print(f"Registered tools: {list(self._tool_manager._tools.keys())}", file=sys.stderr)
    
    def _register_memory_tools(self):
        """Register the 8 explicit memory tools using the papr-memory SDK"""
        
        @self.tool()
        async def add_memory(
            content: str,
            type: str = "text",
            metadata: Optional[Dict[str, Any]] = None,
            context: Optional[List[Dict[str, str]]] = None,
            relationships_json: Optional[List[Dict[str, Any]]] = None,
            skip_background_processing: bool = False
        ) -> Dict[str, Any]:
            """
            Add a new memory item to Papr Memory API.
            
            Args:
                content: The content of the memory item
                type: Type of memory (text, code_snippet, document)
                metadata: Optional metadata for the memory item
                context: Optional context for the memory item
                relationships_json: Optional relationships for Graph DB
                skip_background_processing: Skip background processing if True
                
            Returns:
                Dict containing the added memory item details
            """
            try:
                logger.info(f"Adding memory: {content[:100]}...")
                print(f"Adding memory: {content[:100]}...", file=sys.stderr)
                
                # Prepare memory data
                memory_data = {
                    "content": content,
                    "type": type
                }
                
                if metadata:
                    memory_data["metadata"] = metadata
                if context:
                    memory_data["context"] = context
                if relationships_json:
                    memory_data["relationships_json"] = relationships_json
                
                # Add memory using SDK
                result = papr_client.memory.add(**memory_data)
                
                logger.info(f"Memory added successfully: {result}")
                print(f"Memory added successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error adding memory: {str(e)}")
                print(f"ERROR adding memory: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def get_memory(memory_id: str) -> Dict[str, Any]:
            """
            Retrieve a memory item by ID.
            
            Args:
                memory_id: The ID of the memory item to retrieve
                
            Returns:
                Dict containing the memory item details
            """
            try:
                logger.info(f"Getting memory: {memory_id}")
                print(f"Getting memory: {memory_id}", file=sys.stderr)
                
                result = papr_client.memory.get(memory_id)
                
                logger.info(f"Memory retrieved successfully: {result}")
                print(f"Memory retrieved successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error getting memory: {str(e)}")
                print(f"ERROR getting memory: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def update_memory(
            memory_id: str,
            content: Optional[str] = None,
            type: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            context: Optional[List[Dict[str, str]]] = None,
            relationships_json: Optional[List[Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
            """
            Update an existing memory item.
            
            Args:
                memory_id: The ID of the memory item to update
                content: New content for the memory item
                type: New type for the memory item
                metadata: Updated metadata for the memory item
                context: Updated context for the memory item
                relationships_json: Updated relationships for Graph DB
                
            Returns:
                Dict containing the updated memory item details
            """
            try:
                logger.info(f"Updating memory: {memory_id}")
                print(f"Updating memory: {memory_id}", file=sys.stderr)
                
                # Prepare update data
                update_data = {}
                if content is not None:
                    update_data["content"] = content
                if type is not None:
                    update_data["type"] = type
                if metadata is not None:
                    update_data["metadata"] = metadata
                if context is not None:
                    update_data["context"] = context
                if relationships_json is not None:
                    update_data["relationships_json"] = relationships_json
                
                # Update memory using SDK
                result = papr_client.memory.update(memory_id, **update_data)
                
                logger.info(f"Memory updated successfully: {result}")
                print(f"Memory updated successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error updating memory: {str(e)}")
                print(f"ERROR updating memory: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def delete_memory(memory_id: str, skip_parse: bool = False) -> Dict[str, Any]:
            """
            Delete a memory item by ID.
            
            Args:
                memory_id: The ID of the memory item to delete
                skip_parse: Skip Parse Server deletion if True
                
            Returns:
                Dict containing the deletion result
            """
            try:
                logger.info(f"Deleting memory: {memory_id}")
                print(f"Deleting memory: {memory_id}", file=sys.stderr)
                
                result = papr_client.memory.delete(memory_id, skip_parse=skip_parse)
                
                logger.info(f"Memory deleted successfully: {result}")
                print(f"Memory deleted successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error deleting memory: {str(e)}")
                print(f"ERROR deleting memory: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def search_memory(
            query: str,
            max_memories: int = 20,
            max_nodes: int = 15,
            rank_results: bool = False,
            enable_agentic_graph: bool = False,
            user_id: Optional[str] = None,
            external_user_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Search through memories with authentication required.
            
            Args:
                query: Detailed search query describing what you're looking for
                max_memories: Maximum number of memories to return (recommended: 15-20)
                max_nodes: Maximum number of neo nodes to return (recommended: 10-15)
                rank_results: Whether to enable additional ranking of search results
                enable_agentic_graph: Enable agentic graph search for intelligent results
                user_id: Optional internal user ID to filter search results
                external_user_id: Optional external user ID to filter search results
                metadata: Optional metadata filter
                
            Returns:
                Dict containing search results with memories and nodes
            """
            try:
                logger.info(f"Searching memories: {query[:100]}...")
                print(f"Searching memories: {query[:100]}...", file=sys.stderr)
                
                # Prepare search parameters
                search_params = {
                    "query": query,
                    "max_memories": max_memories,
                    "max_nodes": max_nodes,
                    "rank_results": rank_results,
                    "enable_agentic_graph": enable_agentic_graph
                }
                
                if user_id:
                    search_params["user_id"] = user_id
                if external_user_id:
                    search_params["external_user_id"] = external_user_id
                if metadata:
                    search_params["metadata"] = metadata
                
                # Search memories using SDK
                result = papr_client.memory.search(**search_params)
                
                logger.info(f"Memory search completed successfully")
                print(f"Memory search completed successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error searching memories: {str(e)}")
                print(f"ERROR searching memories: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def submit_feedback(
            search_id: str,
            feedback_type: str,
            feedback_source: str = "inline",
            feedback_text: Optional[str] = None,
            feedback_score: Optional[float] = None,
            feedback_value: Optional[str] = None,
            cited_memory_ids: Optional[List[str]] = None,
            cited_node_ids: Optional[List[str]] = None,
            feedback_processed: Optional[bool] = None,
            feedback_impact: Optional[str] = None,
            user_id: Optional[str] = None,
            external_user_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Submit feedback on search results to help improve model performance.
            
            Args:
                search_id: The search_id from SearchResponse that this feedback relates to
                feedback_type: Type of feedback (thumbs_up, thumbs_down, rating, etc.)
                feedback_source: Source of feedback (inline, external, etc.)
                feedback_text: Optional text feedback
                feedback_score: Optional numerical score
                feedback_value: Optional feedback value
                cited_memory_ids: Optional list of cited memory IDs
                cited_node_ids: Optional list of cited node IDs
                feedback_processed: Whether feedback has been processed
                feedback_impact: Impact of the feedback
                user_id: Optional internal user ID
                external_user_id: Optional external user ID
                
            Returns:
                Dict containing the feedback submission result
            """
            try:
                logger.info(f"Submitting feedback for search: {search_id}")
                print(f"Submitting feedback for search: {search_id}", file=sys.stderr)
                
                # Prepare feedback data
                feedback_data = {
                    "feedbackType": feedback_type,
                    "feedbackSource": feedback_source
                }
                
                if feedback_text:
                    feedback_data["feedbackText"] = feedback_text
                if feedback_score is not None:
                    feedback_data["feedbackScore"] = feedback_score
                if feedback_value:
                    feedback_data["feedbackValue"] = feedback_value
                if cited_memory_ids:
                    feedback_data["citedMemoryIds"] = cited_memory_ids
                if cited_node_ids:
                    feedback_data["citedNodeIds"] = cited_node_ids
                if feedback_processed is not None:
                    feedback_data["feedbackProcessed"] = feedback_processed
                if feedback_impact:
                    feedback_data["feedbackImpact"] = feedback_impact
                
                # Submit feedback using SDK
                result = papr_client.feedback.submit(
                    search_id=search_id,
                    feedback_data=feedback_data,
                    user_id=user_id,
                    external_user_id=external_user_id
                )
                
                logger.info(f"Feedback submitted successfully: {result}")
                print(f"Feedback submitted successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error submitting feedback: {str(e)}")
                print(f"ERROR submitting feedback: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def submit_batch_feedback(
            feedback_items: List[Dict[str, Any]],
            session_context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Submit multiple feedback items in a single request.
            
            Args:
                feedback_items: List of feedback items to submit (max 100)
                session_context: Optional session-level context for batch feedback
                
            Returns:
                Dict containing the batch feedback submission result
            """
            try:
                logger.info(f"Submitting batch feedback: {len(feedback_items)} items")
                print(f"Submitting batch feedback: {len(feedback_items)} items", file=sys.stderr)
                
                # Submit batch feedback using SDK
                result = papr_client.feedback.submit_batch(
                    feedback_items=feedback_items,
                    session_context=session_context
                )
                
                logger.info(f"Batch feedback submitted successfully: {result}")
                print(f"Batch feedback submitted successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error submitting batch feedback: {str(e)}")
                print(f"ERROR submitting batch feedback: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def add_memory_batch(
            memories: List[Dict[str, Any]],
            user_id: Optional[str] = None,
            external_user_id: Optional[str] = None,
            batch_size: int = 10,
            skip_background_processing: bool = False,
            webhook_url: Optional[str] = None,
            webhook_secret: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Add multiple memory items in a batch with size validation and background processing.
            
            Args:
                memories: List of memory items to add in batch (max 50)
                user_id: Internal user ID for all memories in the batch
                external_user_id: External user ID for all memories in the batch
                batch_size: Number of items to process in parallel (default: 10)
                skip_background_processing: Skip background processing if True
                webhook_url: Optional webhook URL to notify when batch processing is complete
                webhook_secret: Optional secret key for webhook authentication
                
            Returns:
                Dict containing the batch add result
            """
            try:
                logger.info(f"Adding memory batch: {len(memories)} items")
                print(f"Adding memory batch: {len(memories)} items", file=sys.stderr)
                
                # Prepare batch parameters
                batch_params = {
                    "memories": memories,
                    "batch_size": batch_size,
                    "skip_background_processing": skip_background_processing
                }
                
                if user_id:
                    batch_params["user_id"] = user_id
                if external_user_id:
                    batch_params["external_user_id"] = external_user_id
                if webhook_url:
                    batch_params["webhook_url"] = webhook_url
                if webhook_secret:
                    batch_params["webhook_secret"] = webhook_secret
                
                # Add memory batch using SDK
                result = papr_client.memory.add_batch(**batch_params)
                
                logger.info(f"Memory batch added successfully: {result}")
                print(f"Memory batch added successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error adding memory batch: {str(e)}")
                print(f"ERROR adding memory batch: {str(e)}", file=sys.stderr)
                raise

def init_mcp():
    """Initialize MCP server with explicit memory tools"""
    try:
        print("Initializing MCP server with explicit tools...", file=sys.stderr)
        
        # Create MCP instance with explicit tools
        mcp = CustomFastMCP(name="Papr Memory MCP")
        
        # Log the tools that were registered
        logger.info(f"Initialized MCP with tools: {list(mcp._tool_manager._tools.keys())}")
        print(f"Initialized MCP with tools: {list(mcp._tool_manager._tools.keys())}", file=sys.stderr)
        return mcp
    except Exception as e:
        logger.error(f"Error initializing MCP: {str(e)}")
        print(f"ERROR initializing MCP: {str(e)}", file=sys.stderr)
        raise

# Try to initialize the MCP with explicit tools
print("Attempting to initialize MCP with explicit tools...", file=sys.stderr)
try:
    mcp = init_mcp()
    print("Successfully initialized MCP with explicit tools", file=sys.stderr)
    print(f"Available tools: {list(mcp._tool_manager._tools.keys())}", file=sys.stderr)
except Exception as e:
    print(f"Failed to initialize MCP with explicit tools: {e}", file=sys.stderr)
    print("Falling back to basic MCP...", file=sys.stderr)
    
    # Fallback to basic MCP if initialization fails
    mcp = FastMCP("Papr Memory MCP")
    
    @mcp.tool()
    async def get_memories(query: str = None) -> str:
        """Get memories from Papr Memory API"""
        return f"Memory search for: {query} (SDK not available)"
    
    @mcp.tool()
    async def add_memory(content: str) -> str:
        """Add a memory to Papr Memory API"""
        return f"Memory added: {content} (SDK not available)"

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
