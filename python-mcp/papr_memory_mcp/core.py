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
from contextvars import ContextVar

# Context variable to store API key per request
request_api_key: ContextVar[str | None] = ContextVar('request_api_key', default=None)

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

# Do not initialize a global Papr client; create per-call with provided api_key

def _get_effective_api_key(passed_api_key: Optional[str]) -> str:
    """Return the api key passed to the tool or fall back to context var or env var.
    Raises a clear error if neither is provided.
    
    Priority:
    1. Explicitly passed api_key parameter
    2. Request-scoped context variable (from Bearer token)
    3. Environment variable PAPR_API_KEY
    """
    if passed_api_key and passed_api_key.strip():
        return passed_api_key
    
    # Check request-scoped context variable (set by middleware from Bearer token)
    ctx_key = request_api_key.get()
    if ctx_key and ctx_key.strip():
        return ctx_key
    
    # Fall back to environment variable
    env_key = os.getenv("PAPR_API_KEY")
    if env_key and env_key.strip():
        return env_key
    
    raise ValueError("API key is required. Pass 'api_key' or set PAPR_API_KEY in environment.")

def install_bearer_middleware(app):
    """Install Bearer token middleware using FastAPI's decorator approach"""
    from fastapi import Request
    
    @app.middleware("http")
    async def bearer_token_middleware(request: Request, call_next):
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            request_api_key.set(auth[7:])
            print(f"[MW] Bearer OK on {request.url.path}", file=sys.stderr)
        else:
            request_api_key.set(None)
            print(f"[MW] No Bearer on {request.url.path}", file=sys.stderr)
        return await call_next(request)

def _ensure_mw(app):
    """Ensure middleware is installed on app (guard against multiple installations)"""
    if app and not getattr(app, "_papr_mw_installed", False):
        install_bearer_middleware(app)
        app._papr_mw_installed = True
        print(f"[MW] Middleware installed on app id={id(app)}", file=sys.stderr)
    return app

class CustomFastMCP(FastMCP):
    def __init__(self, **settings):
        """Initialize CustomFastMCP with Bearer token middleware support"""
        # Keep stateless_http=True for stateless operation on /mcp
        settings.setdefault("stateless_http", True)
        # Do NOT set json_response - let FastMCP support both JSON-RPC and SSE transports
        # based on the Accept header from the client
        
        print("Initializing CustomFastMCP with explicit tools...", file=sys.stderr)
        super().__init__(name="Papr Memory MCP", **settings)
        
        # Register the 8 explicit memory tools
        self._register_memory_tools()
        
        logger.info("CustomFastMCP initialized with explicit memory tools")
        print("CustomFastMCP initialized with explicit memory tools", file=sys.stderr)
        logger.info(f"Registered tools: {list(self._tool_manager._tools.keys())}")
        print(f"Registered tools: {list(self._tool_manager._tools.keys())}", file=sys.stderr)
        
        # Register health endpoint after initialization
        self._register_health_endpoint()
        
        # Add robust error handling for connection issues
        self._setup_robust_error_handling()
    
    # FastMCP 2.x exposes these factories; override each to ensure middleware
    def http_app(self, *args, **kwargs):
        """Override http_app to ensure middleware is installed"""
        app = super().http_app(*args, **kwargs)
        return _ensure_mw(app)
    
    def streamable_http_app(self, *args, **kwargs):
        """Override streamable_http_app to ensure middleware is installed"""
        try:
            app = super().streamable_http_app(*args, **kwargs)
            return _ensure_mw(app)
        except AttributeError:
            # Method might not exist in all FastMCP versions
            return None
    
    def sse_app(self, *args, **kwargs):
        """Override sse_app to ensure middleware is installed"""
        try:
            app = super().sse_app(*args, **kwargs)
            return _ensure_mw(app)
        except AttributeError:
            # Method might not exist in all FastMCP versions
            return None
    
    def _setup_robust_error_handling(self):
        """Setup robust error handling for connection issues"""
        try:
            import asyncio
            from anyio import ClosedResourceError
            
            # Override the message router to handle ClosedResourceError gracefully
            original_message_router = None
            if hasattr(self, '_server') and hasattr(self._server, 'message_router'):
                original_message_router = self._server.message_router
                
                async def robust_message_router(*args, **kwargs):
                    try:
                        return await original_message_router(*args, **kwargs)
                    except ClosedResourceError as e:
                        logger.warning(f"Client disconnected gracefully: {e}")
                        print(f"Client disconnected gracefully: {e}", file=sys.stderr)
                        return  # Gracefully handle disconnection
                    except Exception as e:
                        logger.error(f"Error in message router: {e}")
                        print(f"Error in message router: {e}", file=sys.stderr)
                        # Don't re-raise to prevent server crashes
                        
                self._server.message_router = robust_message_router
                logger.info("Robust error handling configured for message router")
                
        except Exception as e:
            logger.warning(f"Could not setup robust error handling: {e}")
            print(f"Could not setup robust error handling: {e}", file=sys.stderr)
    
    def _validate_connection_protocol(self, request):
        """Validate and handle HTTP vs HTTPS protocol issues"""
        try:
            # Check if request is secure
            is_secure = getattr(request, 'is_secure', False) or \
                       request.headers.get('x-forwarded-proto') == 'https' or \
                       request.headers.get('x-forwarded-ssl') == 'on'
            
            if not is_secure and request.url.scheme == 'http':
                logger.warning("Insecure HTTP connection detected - consider using HTTPS")
                print("Insecure HTTP connection detected - consider using HTTPS", file=sys.stderr)
            
            return is_secure
            
        except Exception as e:
            logger.warning(f"Could not validate connection protocol: {e}")
            return True  # Default to secure if we can't determine
    
    def _register_health_endpoint(self):
        """Register health check endpoint"""
        try:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
            
            # Try different ways to get the FastAPI app
            app = None
            
            # Method 1: Call http_app function (preferred in newer versions)
            if hasattr(self, 'http_app'):
                try:
                    app = self.http_app()
                    logger.info("Got FastAPI app from http_app()")
                except Exception as e:
                    logger.warning(f"Failed to call http_app(): {e}")
            # Method 2: Call streamable_http_app function (fallback)
            elif hasattr(self, 'streamable_http_app'):
                try:
                    app = self.streamable_http_app()
                    logger.info("Got FastAPI app from streamable_http_app()")
                except Exception as e:
                    logger.warning(f"Failed to call streamable_http_app(): {e}")
            # Method 3: Call sse_app function
            elif hasattr(self, 'sse_app'):
                try:
                    app = self.sse_app()
                    logger.info("Got FastAPI app from sse_app()")
                except Exception as e:
                    logger.warning(f"Failed to call sse_app(): {e}")
            # Method 4: Check if we have direct access to app
            elif hasattr(self, '_app') and self._app:
                app = self._app
            # Method 5: Check server attribute
            elif hasattr(self, '_server') and hasattr(self._server, 'app'):
                app = self._server.app
            
            if app is None:
                # Debug: Print available attributes
                logger.warning("Could not find FastAPI app for health endpoint")
                logger.warning(f"Available attributes: {dir(self)}")
                print(f"Available attributes: {dir(self)}", file=sys.stderr)
                return
            
            # Try to register health endpoints using different methods
            try:
                # Method 1: Try using add_route if available
                if hasattr(app, 'add_route'):
                    async def health_check():
                        """Health check endpoint for the MCP server"""
                        return JSONResponse({
                            "status": "healthy",
                            "server": "Papr Memory MCP",
                            "tools": list(self._tool_manager._tools.keys()),
                            "version": "1.0.0",
                            "robustness": {
                                "error_handling": "enabled",
                                "retry_logic": "enabled",
                                "connection_validation": "enabled"
                            }
                        })
                    
                    async def debug_endpoint():
                        """Debug endpoint to test middleware and environment"""
                        return JSONResponse({
                            "status": "debug",
                            "papr_api_key_set": bool(os.getenv("PAPR_API_KEY")),
                            "papr_api_key_preview": os.getenv("PAPR_API_KEY", "NOT_SET")[:8] + "..." if os.getenv("PAPR_API_KEY") else "NOT_SET",
                            "environment_keys": list(os.environ.keys())
                        })
                    
                    app.add_route("/mcp/health", health_check, methods=["GET"])
                    app.add_route("/mcp/debug", debug_endpoint, methods=["GET"])
                    logger.info("Health endpoints registered via add_route")
                    print("Health endpoints registered via add_route", file=sys.stderr)
                # Method 2: Try using get decorator if available
                elif hasattr(app, 'get') and callable(getattr(app, 'get')):
                    @app.get("/mcp/health")
                    async def health_check():
                        """Health check endpoint for the MCP server"""
                        return JSONResponse({
                            "status": "healthy",
                            "server": "Papr Memory MCP",
                            "tools": list(self._tool_manager._tools.keys()),
                            "version": "1.0.0",
                            "robustness": {
                                "error_handling": "enabled",
                                "retry_logic": "enabled",
                                "connection_validation": "enabled"
                            }
                        })
                    
                    @app.get("/mcp/debug")
                    async def debug_endpoint():
                        """Debug endpoint to test middleware and environment"""
                        return JSONResponse({
                            "status": "debug",
                            "papr_api_key_set": bool(os.getenv("PAPR_API_KEY")),
                            "papr_api_key_preview": os.getenv("PAPR_API_KEY", "NOT_SET")[:8] + "..." if os.getenv("PAPR_API_KEY") else "NOT_SET",
                            "environment_keys": list(os.environ.keys())
                        })
                    
                    logger.info("Health endpoints registered via decorator")
                    print("Health endpoints registered via decorator", file=sys.stderr)
                else:
                    logger.warning(f"App object doesn't support route registration. Type: {type(app)}")
                    print(f"App object doesn't support route registration. Type: {type(app)}", file=sys.stderr)
            except Exception as e:
                logger.error(f"Failed to register health endpoints: {e}")
                print(f"Failed to register health endpoints: {e}", file=sys.stderr)
            
        except Exception as e:
            logger.error(f"Failed to register health endpoint: {e}")
            print(f"Failed to register health endpoint: {e}", file=sys.stderr)
    
    
    def _register_memory_tools(self):
        """Register the 8 explicit memory tools using the papr-memory SDK"""
        
        @self.tool()
        async def add_memory(
            content: str,
            api_key: Optional[str] = None,
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
                
                # Use per-call Papr client with provided or env API key
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
                result = papr_client.memory.add(**memory_data)
                
                logger.info(f"Memory added successfully: {result}")
                print(f"Memory added successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error adding memory: {str(e)}")
                print(f"ERROR adding memory: {str(e)}", file=sys.stderr)
                # Return a structured error response instead of raising
                return {
                    "error": True,
                    "message": f"Failed to add memory: {str(e)}",
                    "details": str(e)
                }
        
        @self.tool()
        async def get_memory(memory_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
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
                
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
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
            api_key: Optional[str] = None,
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
                
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
                result = papr_client.memory.update(memory_id, **update_data)
                
                logger.info(f"Memory updated successfully: {result}")
                print(f"Memory updated successfully", file=sys.stderr)
                return result
                
            except Exception as e:
                logger.error(f"Error updating memory: {str(e)}")
                print(f"ERROR updating memory: {str(e)}", file=sys.stderr)
                raise
        
        @self.tool()
        async def delete_memory(memory_id: str, api_key: Optional[str] = None, skip_parse: bool = False) -> Dict[str, Any]:
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
                
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
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
            api_key: Optional[str] = None,
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
                
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
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
            api_key: Optional[str] = None,
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
                
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
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
            api_key: Optional[str] = None,
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
                
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
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
            api_key: Optional[str] = None,
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
                
                effective_key = _get_effective_api_key(api_key)
                papr_client = Papr(x_api_key=effective_key)
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
        
        # Create MCP instance with explicit tools and stateless configuration
        # Middleware is installed automatically via app factory overrides
        mcp = CustomFastMCP()
        
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
    """Main entry point for the Papr MCP server with robust error handling."""
    max_retries = 3
    retry_count = 0
    
    # Detect transport mode from environment or stdin/stdout
    # If running in a pipe/stdio context (like from MCP client), use stdio
    # Otherwise, use HTTP for Docker/Azure deployment
    transport_mode = os.getenv("MCP_TRANSPORT", "auto")
    
    # Auto-detect: if stdin is a terminal, use HTTP; if piped, use stdio
    if transport_mode == "auto":
        import sys
        if sys.stdin.isatty():
            transport_mode = "http"
            print("Auto-detected HTTP transport (terminal mode)", file=sys.stderr)
        else:
            transport_mode = "stdio"
            print("Auto-detected stdio transport (pipe mode)", file=sys.stderr)
    
    logger.info(f"Using transport mode: {transport_mode}")
    print(f"Using transport mode: {transport_mode}", file=sys.stderr)
    
    while retry_count < max_retries:
        try:
            # Start the server
            print("=== STARTING MCP SERVER ===", file=sys.stderr)
            logger.info("Starting MCP server process...")
            logger.info("About to call mcp.run()...")
            print("About to call mcp.run()...", file=sys.stderr)
            
            # Use appropriate transport based on detection
            if transport_mode == "stdio":
                # Use stdio transport for MCP client testing
                print("Starting stdio transport...", file=sys.stderr)
                mcp.run(transport="stdio")
            else:
                # Use HTTP transport for Docker/Azure deployment
                print("Starting HTTP transport...", file=sys.stderr)
                # Get port from environment (Azure sets PORT or WEBSITES_PORT)
                port = int(os.getenv("PORT", os.getenv("WEBSITES_PORT", "8000")))
                print(f"Using port: {port}", file=sys.stderr)
                # Add timeout and keep-alive settings for better connection handling
                mcp.run(
                    transport="http", 
                    host="0.0.0.0", 
                    port=port,
                    # Add connection timeout and keep-alive settings
                    timeout=30,
                    keep_alive=True
                )
            
            print("MCP server finished running", file=sys.stderr)
            logger.info("MCP server finished running")
            break  # Success, exit retry loop
            
        except KeyboardInterrupt:
            print("Received keyboard interrupt, shutting down...", file=sys.stderr)
            logger.info("Received keyboard interrupt, shutting down...")
            break
            
        except Exception as e:
            retry_count += 1
            error_msg = f"ERROR running MCP server (attempt {retry_count}/{max_retries}): {str(e)}"
            print(error_msg, file=sys.stderr)
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...", file=sys.stderr)
                logger.info(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
            else:
                print("Max retries exceeded, giving up", file=sys.stderr)
                logger.error("Max retries exceeded, giving up")
                raise

if __name__ == "__main__":
    main()
