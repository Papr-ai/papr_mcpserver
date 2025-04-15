from typing import Dict, Any, Optional, List
from fastapi import BackgroundTasks, HTTPException
from models.parse_server import AddMemoryResponse, AddMemoryItem
from memory.memory_graph import MemoryGraph
from services.user import User
from amplitude import Amplitude, BaseEvent, EventOptions, Identify
from services.logging_config import get_logger
from services.connector_service import find_user_by_connector_ids
import json
from memory.memory_graph import MemoryGraph
from memory.memory_item import (
    TextMemoryItem, CodeSnippetMemoryItem, DocumentMemoryItem,
    WebpageMemoryItem, CodeFileMemoryItem, MeetingMemoryItem,
    PluginMemoryItem, IssueMemoryItem, CustomerMemoryItem
)
from models.parse_server import UpdateMemoryResponse,  AddMemoryResponse, AddMemoryItem, BatchMemoryError, BatchMemoryResponse
from os import environ as env
from dotenv import find_dotenv, load_dotenv
from redis.asyncio import Redis
from services.logging_config import get_logger
from urllib.parse import urlparse
import uuid

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

logger = get_logger(__name__)

# Log at module level to verify logger is working
logger.info("Memory routes module loaded")

# Initialize Amplitude client
amplitude_client = Amplitude(env.get("AMPLITUDE_API_KEY"))
logger.info(f"Amplitude client initialized with API key: {env.get('AMPLITUDE_API_KEY')}")

# Get API key for hotglue
hotglue_api_key = env.get("HOTGLUE_PAPR_API_KEY")
logger.info(f"hotglue_api_key inside memory_routes.py: {hotglue_api_key}")

# Initialize chat_gpt
chat_gpt = None  # This should be properly initialized based on your application's needs


async def handle_incoming_memory(
    data: Dict[str, Any],
    user_id: str,
    sessionToken: str,
    user_info: Optional[Dict[str, Any]],
    client_type: str,
    memory_graph: MemoryGraph,
    background_tasks: BackgroundTasks,
    skip_background_processing: bool = False,
    user_workspace_ids: Optional[List[str]] = None
) -> AddMemoryResponse:
    """
    Handle a single memory item addition.
    Returns an AddMemoryResponse.
    """
    try:
        # Check memory limits
        user = User(user_id)
        limit_check = await user.check_memory_limits()
        if limit_check is not None:
            error_response, status_code = limit_check
            if isinstance(error_response, dict):
                raise HTTPException(
                    status_code=status_code, 
                    detail=error_response.get("message", error_response.get("error", "Unknown error"))
                )
            else:
                raise HTTPException(status_code=status_code, detail=str(error_response))

        # Identify user in Amplitude
        try:
            identify = Identify()
            identify.set('user id', user_id)
            amplitude_client.identify(identify, EventOptions(user_id=user_id))
        except Exception as e:
            logger.error(f"Error identifying user in Amplitude: {e}")
            # Continue processing even if Amplitude tracking fails
            
        content = data.get('content')
        memory_type = data.get('type')
        logger.info(f"memory_type: {memory_type}")
        
        metadata = data.get('metadata', {})
        logger.info(f"Type of metadata handle_add_memory: {type(metadata)}")

        # Add a log before extracting memory_type_metadata
        logger.info("Attempting to extract 'type' from metadata.")
        
        memory_type_metadata = metadata.get('type')
        logger.info(f"memory_type inside metadata: {memory_type_metadata}")

        is_private = metadata.get('is_private', True)
        logger.info(f"is_private: {is_private}")

        # If memory_type is 'message', adjust user_id and sessionToken
        if memory_type_metadata == 'message':
            connector = metadata.get('connector')
            connector_user_id = metadata.get('user')
            logger.info(f"Connector user ID from metadata: {connector_user_id}")

            if connector and connector_user_id:
                # Retrieve ACL object IDs using the connector_service
                acl_object_ids = await find_user_by_connector_ids(sessionToken, connector, [connector_user_id])
                logger.info(f"ACL Object IDs: {acl_object_ids}")

                if acl_object_ids:
                    real_user_id = acl_object_ids[0]
                    logger.info(f"Real user ID: {real_user_id}")

                    # Lookup session token for the real user
                    real_sessionToken = await User.lookup_user_token(real_user_id)
                    logger.info(f"Real sessionToken: {real_sessionToken}")

                    # Verify the session token
                    real_user_info = await User.verify_session_token(real_sessionToken)
                    logger.info(f"Verified real user info: {real_user_info}")

                    if real_user_info:
                        user_id = real_user_id
                        sessionToken = real_sessionToken
                        logger.info("Attributed memory to the actual message creator.")
                    else:
                        logger.error("Session token verification failed for real user.")
                        # Retain original user_id and sessionToken (Slack admin)
                        logger.info("Using original user_id and sessionToken (Slack admin).")
                else:
                    logger.warning("No ACL Object IDs found for connector user.")
                    # Retain original user_id and sessionToken (Slack admin)
                    logger.info("Using original user_id and sessionToken (Slack admin) because message creator doesn't have a Papr account.")
            else:
                logger.error("Missing connector or user ID in metadata.")
                # Retain original user_id and sessionToken (Slack admin)
                logger.info("Using original user_id and sessionToken (Slack admin) due to missing connector or user ID.")

        additional_user_ids = metadata.get('additional_user_ids', [])
        context = data.get('context')
        relationships_json = data.get('relationships_json', [])
        project_id = data.get('project_id')
        workspace_id = metadata.get('workspace_id')
        logger.info(f"workspace_id: {workspace_id}")
        sourceType = metadata.get('sourceType')
        logger.info(f"sourceType: {sourceType}")
        sourceUrl = metadata.get('sourceUrl')
        logger.info(f"sourceUrl: {sourceUrl}")
        postMessageId = metadata.get('postMessageId')
        logger.info(f"postMessageId: {postMessageId}")
        pageId = metadata.get('pageId')
        logger.info(f"pageId: {pageId}")
        url = data.get('url')
        logger.info(f"url: {url}")
        connector = metadata.get('connector')
        logger.info(f"connector: {connector}")

        if not metadata:
            metadata = {}

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata format")

        if not relationships_json:
            relationships_json = []

        if isinstance(relationships_json, str):
            try:
                relationships_json = json.loads(relationships_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid relationships_json format")
            
        if not isinstance(relationships_json, list):
            raise HTTPException(status_code=400, detail="relationships_json should be a list")

        metadata['user_id'] = str(user_id)
        metadata['workspace_id'] = str(workspace_id)

        # Handle ACL
        tenant_id = data.get('tenant_id')
        acl_object_ids = metadata.get('acl_object_ids', [])

        if not additional_user_ids:
            additional_user_ids = acl_object_ids

        logger.info(f"additional_user_ids in handle_add_memory: {additional_user_ids}")

        if workspace_id or tenant_id:
            if is_private is True:
                user_acl = {user_id: {"read": True, "write": True}}
                acl_private = User.transform_acl(user_acl, workspace_id, user_id, additional_user_ids)
                if acl_private:
                    metadata['user_read_access'] = acl_private['user_read_access']
                    metadata['user_write_access'] = acl_private['user_write_access']
                    metadata['workspace_read_access'] = []
                    metadata['workspace_write_access'] = []
                    metadata['role_read_access'] = []
                    metadata['role_write_access'] = []
            else:
                acl = None
                if postMessageId:
                    acl = await User.get_acl_for_postMessage(workspace_id=workspace_id, post_message_id=postMessageId, user_id=user_id, additional_user_ids=additional_user_ids)
                elif pageId: 
                    acl = await User.get_acl_for_post(workspace_id=workspace_id, page_id=pageId, user_id=user_id, additional_user_ids=additional_user_ids)
                elif connector:
                    acl = await User.get_acl_for_workspace(workspace_id=workspace_id, tenant_id=tenant_id, user_id=user_id, additional_user_ids=additional_user_ids)

                if acl:
                    metadata['user_read_access'] = acl['user_read_access']
                    metadata['user_write_access'] = acl['user_write_access']
                    metadata['workspace_read_access'] = acl['workspace_read_access']
                    metadata['workspace_write_access'] = acl['workspace_write_access']
                    metadata['role_read_access'] = acl['role_read_access']
                    metadata['role_write_access'] = acl['role_write_access']
                else:
                    # Default to private access if ACL retrieval fails
                    metadata['user_read_access'] = [user_id]
                    metadata['user_write_access'] = [user_id]
                    metadata['workspace_read_access'] = []
                    metadata['workspace_write_access'] = []
                    metadata['role_read_access'] = []
                    metadata['role_write_access'] = []
        else: 
            metadata['user_read_access'] = [user_id]
            metadata['user_write_access'] = [user_id]
            metadata['workspace_read_access'] = []
            metadata['workspace_write_access'] = []
            metadata['role_read_access'] = []
            metadata['role_write_access'] = []

        metadata['pageId'] = str(pageId)
        metadata['sourceType'] = str(sourceType)
        metadata['sourceUrl'] = str(sourceUrl)

        logger.info(f"metadata in handle_add_memory: {metadata}")

        # Create the appropriate MemoryItem instance based on the types
        if memory_type in ['text', 'TextMemoryItem', 'message']:  # Add 'message' to accepted types
            memory_item = TextMemoryItem(content, metadata, context)
        elif memory_type == 'code_snippet': 
            memory_item = CodeSnippetMemoryItem(content, metadata, context)
        elif memory_type == 'DocumentMemoryItem':  # Use consistent type string
            metadata['url'] = url
            metadata['sourceType'] = 'papr'
            memory_item = DocumentMemoryItem(content, metadata, context)
        elif memory_type == 'webpage':
            memory_item = WebpageMemoryItem(content, metadata, context)
        elif memory_type == 'code_file':
            memory_item = CodeFileMemoryItem(content, metadata, context)
        elif memory_type == 'meeting':
            memory_item = MeetingMemoryItem(content, metadata, context)
        elif memory_type == 'plugin':
            memory_item = PluginMemoryItem(content, metadata, context)
        elif memory_type == 'issue':
            memory_item = IssueMemoryItem(content, metadata, context)
        elif memory_type == 'customer':
            memory_item = CustomerMemoryItem(content, metadata, context)
        else:
            raise HTTPException(status_code=400, detail="Invalid memory type")

        logger.info(f"memory_item: {memory_item}")
        
        # Add memory item to graph
        memory_items = await memory_graph.add_memory_item_async(
            memory_item,
            relationships_json,
            sessionToken,
            user_id,
            background_tasks,
            True,
            workspace_id,
            skip_background_processing,
            user_workspace_ids
        )
        logger.info(f"handle_incoming_memory - Raw memory_items: {memory_items}")
        logger.info(f"handle_incoming_memory - memory_items type: {type(memory_items)}")

        if not memory_items:
            raise HTTPException(status_code=404, detail="There was an error adding the memory item")

        # Use the first memory item for logging and response
        first_memory = memory_items[0] if memory_items else None
        logger.info(f"handle_incoming_memory - First item memoryChunkIds: {first_memory.memoryChunkIds if first_memory else 'No memory items'}")

        if user_info:
            # Extracting the required information
            email_address = user_info.get('email')
            geo_info = user_info.get('geoip', {})

            # Construct event_properties
            event_properties = {
                'client_type': client_type,
                'email': email_address
            }

            # Add geo info if available
            for key in ['country_code', 'country_name', 'city_name', 'time_zone', 'continent_code', 'subdivision_code', 'subdivision_name']:
                if key in geo_info and geo_info[key] is not None:
                    event_properties[key] = geo_info[key]

            # Convert latitude and longitude to float if present
            for key in ['latitude', 'longitude']:
                if key in geo_info and geo_info[key] is not None:
                    try:
                        event_properties[key] = float(geo_info[key])
                    except ValueError:
                        logger.warning(f"Omitting {key} due to invalid format: {geo_info[key]}")

            try:
                amplitude_client.track(
                    BaseEvent(
                        event_type="add_memory",
                        user_id=user_id,
                        event_properties=event_properties
                    )
                )
            except Exception as e:
                logger.error(f"Error tracking event in Amplitude: {e}")
        else:
            try:
                amplitude_client.track(
                    BaseEvent(
                        event_type="add_memory",
                        user_id=user_id
                    )
                )
            except Exception as e:
                logger.error(f"Error tracking minimal event in Amplitude: {e}")
          
        # Return the memory item data in AddMemoryResponse format
        return AddMemoryResponse(
            code=200,
            status="success",
            data=[
                AddMemoryItem(
                    memoryId=item.memoryId,
                    createdAt=item.createdAt,
                    objectId=item.objectId,
                    memoryChunkIds=item.memoryChunkIds
                ) for item in memory_items
            ]
        )

    except Exception as e:
        logger.error(f"Error processing memory item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
