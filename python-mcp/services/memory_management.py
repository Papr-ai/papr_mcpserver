import requests
import os
import json
from dotenv import find_dotenv, load_dotenv
from os import environ as env
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed  # Ensure as_completed is imported
from memory.memory_item import MemoryItem
from services.logging_config import get_logger
from services.url_utils import clean_url
from models.parse_server import (
    ParseStoredMemory, MemoryRetrievalResult, ParseUserPointer, UpdateMemoryItem, UpdateMemoryResponse, 
    SystemUpdateStatus, ErrorResponse, MemoryParseServer, ParsePointer, MemoryParseServerUpdate, DocumentUploadStatus, DocumentUploadStatusResponse
)
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, BackgroundTasks, Depends
import httpx
from pydantic import ValidationError
from models.parse_server import DocumentUploadStatusType

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Initialize Parse client
PARSE_SERVER_URL = clean_url(env.get("PUBLIC_SERVER_URL"))
PARSE_APPLICATION_ID = clean_url(env.get("PARSE_APPLICATION_ID"))
PARSE_REST_API_KEY = clean_url(env.get("PARSE_REST_API_KEY"))
PARSE_MASTER_KEY = clean_url(env.get("PARSE_MASTER_KEY"))

def create_usecase(user_id: str, session_token: str, name: str, description: str) -> dict:
    url = f"{PARSE_SERVER_URL}/parse/classes/Usecase"
    data = {
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
        "name": name,
        "description": description
    }
    HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
    }
    response = requests.post(url, headers=HEADERS, data=json.dumps(data))
    if response.status_code != 201:
        logger.error(f"Failed to create usecase: {response.text}")
        return None
    return response.json()

def add_list_of_usecases(user_id: str, session_token: str, use_cases: List[dict]):
    for use_case in use_cases:
        if not use_case.get('name') or not use_case.get('description'):
            logger.info(f"Invalid use case: {use_case}")
            continue
        response = create_usecase(user_id, session_token, use_case['name'], use_case['description'])
        if response is not None:
            logger.info(f"Use case added successfully: {use_case['name']}")
        else:
            logger.info(f"Failed to add use case: {use_case['name']}")


def create_goal(user_id: str, session_token: str, title: str, description: str=None, key_results=None):
    url = f"{PARSE_SERVER_URL}/parse/classes/Goal"
    data = {
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
        "title": title
    }

    # Add description and keyResults to data dictionary if they are provided
    if description is not None:
        data["description"] = description
    else:
        data["description"] = ""  # or any default value you see fit
    
    if key_results is not None:
        data["keyResults"] = key_results
    else:
        data["keyResults"] = []  # or any default value you see fit


    HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
    }
    response = requests.post(url, headers=HEADERS, data=json.dumps(data))
    if response.status_code != 201:
        logger.error(f"Failed to create goal: {response.text}")
        return None
    return response.json()

def add_list_of_goals(user_id: str, session_token: str, goals: List[dict]):
    for goal in goals:
        key_results = goal.get('key_results', [])
        description = goal.get('description', [])
        title = goal.get('title', [])
        response = create_goal(user_id, session_token, title, description, key_results)
        if response is not None:
            logger.info(f"Goal added successfully: {goal['title']}")
        else:
            logger.info(f"Failed to add goal: {goal['title']}")


def create_memory_graph_node(user_id: str, session_token: str, name: str) -> str:
    # Endpoint for querying MemoryGraphNode
    query_url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraphNode"
    
    # Query to check if a MemoryGraphNode with the same name and user_id already exists
    query_params = {
        "where": json.dumps({
            "name": name,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        })
    }
    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Send the GET request to check for existing MemoryGraphNode
    response = requests.get(query_url, headers=HEADERS, params=query_params)
    
    # Check the response
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            # If a node with the same name exists, return its objectId
            logger.info(f"MemoryGraphNode with name '{name}' already exists.")
            return data['results'][0]['objectId']
    else:
        logger.error(f"Failed to query memory graph node: {response.text}")
        return None
    
    # If no existing node is found, create a new MemoryGraphNode
    create_url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraphNode"
    data = {
        "name": name,
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
    }
    
    # Send the POST request to create a new MemoryGraphNode
    response = requests.post(create_url, headers=HEADERS, data=json.dumps(data))
    
    # Check the response
    if response.status_code != 201:
        logger.error(f"Failed to create memory graph node: {response.text}")
        return None
    
    # Return the objectId of the new MemoryGraphNode
    return response.json()['objectId']

def create_memory_graph_relationship(user_id: str, session_token: str, name: str) -> str:
    # Endpoint for querying MemoryGraphRelationship
    query_url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraphRelationship"
    
    # Query to check if a MemoryGraphRelationship with the same name and user_id already exists
    query_params = {
        "where": json.dumps({
            "name": name,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        })
    }
    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Send the GET request to check for existing MemoryGraphRelationship
    response = requests.get(query_url, headers=HEADERS, params=query_params)
    
    # Check the response
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            # If a relationship with the same name exists, return its objectId
            logger.info(f"MemoryGraphRelationship with name '{name}' already exists.")
            return data['results'][0]['objectId']
    else:
        logger.error(f"Failed to query memory graph relationship: {response.text}")
        return None
    
    # If no existing relationship is found, create a new MemoryGraphRelationship
    create_url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraphRelationship"
    data = {
        "name": name,
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        }
    }
    
    # Send the POST request to create a new MemoryGraphRelationship
    response = requests.post(create_url, headers=HEADERS, data=json.dumps(data))
    
    # Check the response
    if response.status_code != 201:
        logger.error(f"Failed to create memory graph relationship: {response.text}")
        return None
    
    # Return the objectId of the new MemoryGraphRelationship
    return response.json()['objectId']

def create_memory_graph(user_id: str, session_token: str, schema: dict, nodes: List[dict], relationships: List[dict]) -> str:
    # First, create MemoryGraphNodes and MemoryGraphRelationships
    node_ids = [create_memory_graph_node(user_id, session_token, node['name']) for node in nodes]
    relationship_ids = [create_memory_graph_relationship(user_id, session_token, rel['name']) for rel in relationships]

    # Endpoint for creating a new MemoryGraph
    url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraph"
    
    # Data for the new MemoryGraph
    data = {
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
        "schema": schema,
        "nodes": [
            {
                "__type": "Pointer",
                "className": "MemoryGraphNode",
                "objectId": node_id
            } for node_id in node_ids if node_id is not None
        ],
        "relationships": [
            {
                "__type": "Pointer",
                "className": "MemoryGraphRelationship",
                "objectId": relationship_id
            } for relationship_id in relationship_ids if relationship_id is not None
        ]
    }
    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Send the POST request to create a new MemoryGraph
    response = requests.post(url, headers=HEADERS, data=json.dumps(data))
    
    # Check the response
    if response.status_code != 201:
        logger.error(f"Failed to create memory graph: {response.text}")
        return None
    
    # Return the objectId of the new MemoryGraph
    return response.json()['objectId']
# ... The rest of the CRUD functions for MemoryGraphNode and MemoryGraphRelationship would follow the same pattern ...

def get_user_usecases(user_id: str, session_token: str) -> List[dict]:
    # Endpoint for fetching Usecases
    url = f"{PARSE_SERVER_URL}/parse/classes/Usecase"
    
    # Query parameters with sorting and limit
    params = {
        "where": json.dumps({
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }),
        "keys": "name,description,createdAt",  # Include createdAt
        "order": "-createdAt",  # Sort by createdAt in descending order
        "limit": 10  # Limit to 10 results
    }

    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Send the GET request to fetch Usecases
    response = requests.get(url, headers=HEADERS, params=params)
    
    # Check the response
    if response.status_code != 200:
        logger.error(f"Failed to fetch usecases: {response.text}")
        return None
    
    # Return the list of usecases
    return response.json().get('results', [])

def extract_usecases(existing_usecases: List[dict]) -> List[dict]:
    """Extract usecase names and descriptions from the list of usecases."""
    # Initialize an empty list to hold the usecases
    usecases = []

    # Check if existing_usecases is None
    if existing_usecases is None:
        logger.info("No usecases found. The usecases parameter is None.")
        return usecases  # Return an empty list

    # Sort usecases by createdAt date (assuming it exists in the data)
    sorted_usecases = sorted(existing_usecases, key=lambda x: x.get('createdAt', ''), reverse=True)

    # Take only the first 10 usecases
    for usecase in sorted_usecases[:10]:
        # Extract name and description
        name = usecase.get('name')
        description = usecase.get('description')
        
        if name:  # Only add if name exists
            usecase_info = {
                'name': name,
                'description': description or '',  # Use empty string if description is None
                'status': 'existing'  # Mark as existing usecase
            }
            usecases.append(usecase_info)
    
    return usecases

def extract_goal_titles(existing_goals: List[dict]) -> List[str]:
    # Initialize an empty list to hold the goal titles
    goal_titles = []

    # Check if existing_goals is None
    if existing_goals is None:
        logger.info("No goals found. The existing_goals parameter is None.")
        return goal_titles  # Return an empty list

    # Sort goals by createdAt date (assuming it exists in the data)
    sorted_goals = sorted(existing_goals, key=lambda x: x.get('createdAt', ''), reverse=True)

    # Take only the first 10 goals
    for goal in sorted_goals[:10]:
        goal_titles.append(goal['title'])
    return goal_titles


def get_user_goals(user_id: str, session_token: str) -> List[dict]:
    # Endpoint for fetching Goals
    url = f"{PARSE_SERVER_URL}/parse/classes/Goal"
    
    # Query parameters to filter goals by user and sort by createdAt
    params = {
        "where": json.dumps({
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }),
        "keys": "title,description,createdAt",  # Include createdAt
        "order": "-createdAt",  # Sort by createdAt in descending order
        "limit": 10  # Limit to 10 results
    }

    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Send the GET request to fetch Goals
    response = requests.get(url, headers=HEADERS, params=params)
    
    # Check the response status
    if response.status_code != 200:
        logger.error(f"Failed to fetch goals: {response.text}")
        return None
    
    # Return the list of goals on successful fetch
    return response.json().get('results', [])

def get_user_memGraph_schema(user_id: str, session_token: str) -> dict:
    # Endpoint for fetching MemoryGraph
    url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraph"
    
    # Query parameters to filter by user and sort by createdAt
    params = {
        "where": json.dumps({
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }),
        "order": "-createdAt"  # Sort by createdAt in descending order
    }

    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        logger.error(f"Failed to fetch memory graph schema: {response.text}")
        return None

    data = response.json().get('results', [])

    # Temporary storage for nodes and relationships to deduplicate them
    temp_nodes = {}
    temp_relationships = {}
    
    for item in data:
        schema = item.get('schema', {})
        nodes = schema.get('nodes', [])
        relationships = schema.get('relationships', [])
        
        for node in nodes:
            # Use node name as the key for deduplication
            temp_nodes[node['name']] = node
        
        for relationship in relationships:
            # Use relationship type as the key for deduplication
            rel_type = relationship.get('type') or relationship.get('relation_type')
            if rel_type:
                temp_relationships[rel_type] = relationship
            else:
                logger.warning(f"Relationship without type found: {relationship}")
   
    # Convert to lists and limit to 10 items each
    nodes_list = list(temp_nodes.values())
    relationships_list = list(temp_relationships.values())
    
    # Sort by name/type to ensure consistent results
    nodes_list.sort(key=lambda x: x.get('name', ''))
    relationships_list.sort(key=lambda x: x.get('type', '') or x.get('relation_type', ''))
    
    combined_schema = {
        'nodes': nodes_list[:10],  # Limit to 10 nodes
        'relationships': relationships_list[:10]  # Limit to 10 relationships
    }
    
    return combined_schema

def extract_node_names(nodes: List[dict]) -> List[str]:
    """
    Extracts the names of nodes from the list of node dictionaries.

    Parameters:
    - nodes (list of dict): List of node dictionaries, each containing a 'name' key.

    Returns:
    - list of str: List of node names.
    """
    return [node['name'] for node in nodes]

def extract_relationship_types(relationships: List[dict]) -> List[str]:
    """
    Extracts the types of relationships from the list of relationship dictionaries,
    preferring the 'name' key if available, otherwise the 'type' key.

    Parameters:
    - relationships (list of dict): List of relationship dictionaries, each containing either a 'name' or 'type' key.

    Returns:
    - list of str: List of relationship types.
    """
    extracted_types = []
    for relationship in relationships:
        # Check if 'name' key exists and use it; otherwise, fall back to 'type' if it exists
        if 'name' in relationship:
            extracted_types.append(relationship['name'])
        elif 'type' in relationship:
            extracted_types.append(relationship['type'])
        # Optional: Handle the case where neither 'name' nor 'type' exists
        # else:
        #     extracted_types.append('Unknown')  # Or any other default value or handling logic
    return extracted_types

def convert_acl(metadata: dict) -> dict:
    """Convert custom ACL to Parse ACL format."""
    acl = {}
    
    # Handle user read/write access
    user_read = metadata.get('user_read_access', [])
    user_write = metadata.get('user_write_access', [])
    
    if isinstance(user_read, str):
        user_read = json.loads(user_read.replace("'", '"'))
    if isinstance(user_write, str):
        user_write = json.loads(user_write.replace("'", '"'))
    
    for user_id in set(user_read + user_write):
        acl[user_id] = {
            'read': user_id in user_read,
            'write': user_id in user_write
        }
    
    # Handle role read/write access
    role_read = metadata.get('role_read_access', [])
    role_write = metadata.get('role_write_access', [])
    
    if isinstance(role_read, str):
        role_read = json.loads(role_read.replace("'", '"'))
    if isinstance(role_write, str):
        role_write = json.loads(role_write.replace("'", '"'))
    
    for role in set(role_read + role_write):
        role_key = f'role:{role}'
        acl[role_key] = {
            'read': role in role_read,
            'write': role in role_write
        }
    
    # Handle workspace read/write access
    workspace_read = metadata.get('workspace_read_access', [])
    workspace_write = metadata.get('workspace_write_access', [])
    
    if isinstance(workspace_read, str):
        workspace_read = json.loads(workspace_read.replace("'", '"'))
    if isinstance(workspace_write, str):
        workspace_write = json.loads(workspace_write.replace("'", '"'))
    
    for workspace in set(workspace_read + workspace_write):
        workspace_key = f'workspace:{workspace}'
        acl[workspace_key] = {
            'read': workspace in workspace_read,
            'write': workspace in workspace_write
        }
    
    return acl


async def update_memory_item_parse(
    session_token: str, 
    object_id: str, 
    update_data: Dict[str, Any]
) -> bool:
    """
    Asynchronously updates an existing memory item in Parse Server.

    Args:
        session_token (str): The session token for authentication.
        object_id (str): The ObjectId of the memory item in Parse Server.
        update_data (Dict[str, Any]): The data to update.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Memory/{object_id}"
    
    try:
        # Remove empty lists and None values before validation
        update_data = {k: v for k, v in update_data.items() 
                      if v is not None and (not isinstance(v, list) or len(v) > 0)}
        
        # Fix workspace pointer format if present
        if 'workspace' in update_data and isinstance(update_data['workspace'], dict):
            update_data['workspace'] = {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": update_data['workspace'].get('objectId')
            }

        # Validate update data using MemoryParseServerUpdate
        parse_memory = MemoryParseServerUpdate(**update_data)
        
        # Get validated data, excluding None values and empty lists
        validated_data = {
            k: v for k, v in parse_memory.model_dump(exclude={'createdAt', 'updatedAt'}).items()
            if v is not None and (not isinstance(v, list) or len(v) > 0)
        }

        logger.info(f"Data to update in Parse Server inside update_memory_item_parse: {validated_data}")

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url, 
                headers=headers, 
                json=validated_data,
                timeout=30.0
            )
            
            response.raise_for_status()
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully updated memory item with ObjectId: {object_id}")
                return True
            else:
                logger.error(f"Failed to update memory item: {response.text}")
                return False
                
    except ValidationError as ve:
        logger.error(f"Validation error with update data: {ve}")
        return False
    except httpx.HTTPError as e:
        logger.error(f"HTTP error updating memory item: {e}")
        return False
    except Exception as e:
        logger.error(f"Error updating memory item: {e}")
        return False

async def store_memory_item(user_id: str, session_token: str, memory_item: MemoryItem) -> Optional[ParseStoredMemory]:
    """
    Asynchronously store a memory item based on its type.
    """
    if memory_item.type == "IssueMemoryItem":
        return await store_issue_memory_item(user_id, session_token, memory_item)
    else:
        return await store_generic_memory_item(user_id, session_token, memory_item)

def convert_comma_string_to_list(value: Union[str, List, None]) -> List[str]:
    """Convert a comma-separated string to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    return []

async def store_generic_memory_item(
    user_id: str, 
    session_token: str, 
    memory_item: MemoryItem
) -> Optional[ParseStoredMemory]:
    # At the start of the function
    logger.info(f"Entering store_generic_memory_item with memory_item metadata: {memory_item.metadata}")
    logger.info(f"session_token: {session_token}")
    
    # Define the fields we want Parse to return based on ParseStoredMemory model
    fields_param = (
        "objectId,createdAt,updatedAt,ACL,content,metadata,sourceType,context,"
        "title,location,emojiTags,hierarchicalStructures,type,sourceUrl,"
        "conversationId,memoryId,topics,steps,current_step,memoryChunkIds,"
        "user.objectId,user.displayName,user.fullname,user.profileimage,user.title,user.isOnline,"
        "workspace,post,postMessage"
    )
    url = f"{PARSE_SERVER_URL}/parse/classes/Memory?keys={fields_param}"
    logger.info(f"URL: {url}")

    # Extract metadata
    metadata = memory_item.metadata
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    
    # After extracting metadata
    logger.info(f"Extracted metadata: {metadata}")
    
    # Ensure memoryChunkIds is a proper array
    memory_chunk_ids = metadata.get('memoryChunkIds', [])
    if isinstance(memory_chunk_ids, str):
        try:
            memory_chunk_ids = json.loads(memory_chunk_ids)
        except json.JSONDecodeError:
            memory_chunk_ids = [id.strip() for id in memory_chunk_ids.strip('[]').split(',') if id.strip()]
    
    # After processing memoryChunkIds
    logger.info(f"Processed memoryChunkIds from metadata: {memory_chunk_ids}")
    logger.info(f"Type of memoryChunkIds: {type(memory_chunk_ids)}")
    
    # Ensure memory_chunk_ids is a list of strings
    if isinstance(memory_chunk_ids, list):
        memory_chunk_ids = [str(id).strip() for id in memory_chunk_ids if id]
    else:
        memory_chunk_ids = []
    
    metadata['memoryChunkIds'] = memory_chunk_ids
    logger.info(f"memoryChunkIds in metadata in store_generic_memory_item: {memory_chunk_ids}")

    memory_id_str = str(memory_item.id)
    logger.info(f"Memory item ID: {memory_id_str}")
    context = "" if memory_item.context is None else str(memory_item.context)
    logger.info(f"Context: {context}")
    # Convert custom ACL to Parse ACL
    parse_acl = convert_acl(metadata)

    try:
       # Pre-process list fields to ensure proper format before Pydantic validation
        emoji_tags = convert_comma_string_to_list(
            metadata.get('emojiTags') or 
            metadata.get('emoji_tags') or 
            metadata.get('emoji tags')
        )
        emotion_tags = convert_comma_string_to_list(
            metadata.get('emotionTags') or 
            metadata.get('emotion_tags') or 
            metadata.get('emotion tags')
        )
        topics = convert_comma_string_to_list(metadata.get('topics'))
        steps = convert_comma_string_to_list(metadata.get('steps'))

        # Create base arguments dictionary
        parse_memory_args = {
            "ACL": parse_acl,
            "user": ParsePointer(
                __type="Pointer",
                className="_User",
                objectId=user_id
            ),
            "content": memory_item.content,
            "sourceType": metadata.get('sourceType', ""),
            "context": context,
            "title": metadata.get("title", ""),
            "location": metadata.get("location", ""),
            "emojiTags": emoji_tags,
            "emotionTags": emotion_tags,
            "hierarchicalStructures": (
                metadata.get('hierarchicalStructures') or 
                metadata.get('hierarchical_structures') or 
                metadata.get('hierarchical structures') or 
                ""
            ),
            "type": memory_item.type,
            "sourceUrl": metadata.get('sourceUrl', ""),
            "conversationId": metadata.get("conversationId", ""),
            "memoryId": memory_id_str,
            "topics": topics,
            "steps": steps,
            "current_step": metadata.get('current_step'),
            "memoryChunkIds": memory_chunk_ids,
            "metrics": metadata.get('metrics', {})
        }

        # Add document-specific fields only if type is DocumentMemoryItem
        if memory_item.type == "DocumentMemoryItem":
            document_fields = {
                "page_number": metadata.get('page_number'),
                "total_pages": metadata.get('total_pages'),
                "upload_id": metadata.get('upload_id'),
                "filename": metadata.get('filename'),
                "page": metadata.get('page'),
                "file_url": metadata.get('file_url')
            }
            # Use file_url as sourceUrl if sourceUrl is empty string and file_url exists
            if parse_memory_args["sourceUrl"].strip() == "" and document_fields.get("file_url"):
                parse_memory_args["sourceUrl"] = document_fields["file_url"]
            parse_memory_args.update(document_fields)

        # Create MemoryParseServer object with all fields
        parse_memory = MemoryParseServer(**parse_memory_args)

        logger.info(f"parse_memory: {parse_memory}")

        # Add workspace pointer if available
        workspace_id = metadata.get("workspace_id")
        if workspace_id and workspace_id.strip():
            parse_memory.workspace = ParsePointer(
                __type="Pointer",
                className="WorkSpace",
                objectId=workspace_id
            )

        # Add post pointer if available
        page_id = metadata.get("pageId")
        if page_id and str(page_id).lower() != "none" and page_id.strip():
            parse_memory.post = ParsePointer(
                __type="Pointer",
                className="Post",
                objectId=page_id
            )

        # Add postMessage pointer if available
        post_message_id = metadata.get("postMessageId")
        if post_message_id and str(post_message_id).lower() != "none" and post_message_id.strip():
            parse_memory.postMessage = ParsePointer(
                __type="Pointer",
                className="PostMessage",
                objectId=post_message_id
            )

        # Before creating parse_memory
        logger.info(f"Creating ParseMemoryServer with memoryChunkIds: {memory_chunk_ids}")
        
        # Convert to dict and remove None values
        logger.info(f"parse_memory before model_dump: {parse_memory}")
        data = parse_memory.model_dump(exclude_none=True)
        logger.info(f"data after model_dump: {data}")
        
        # After creating parse_memory
        logger.info(f"Created parse_memory object with memoryChunkIds: {parse_memory.memoryChunkIds}")
        
        # Before converting to dict
        logger.info(f"Final parse_memory object before dict conversion: {parse_memory}")
        
        # Log the final data structure
        logger.info(f"Final data structure before sending to Parse Server: {json.dumps(data, indent=2)}")
        logger.info(f"Type of memoryChunkIds in data: {type(data.get('memoryChunkIds'))}")
        logger.info(f"Value of memoryChunkIds in data: {data.get('memoryChunkIds')}")
        logger.info(f"Actual JSON string for memoryChunkIds: {json.dumps(data.get('memoryChunkIds'))}")

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=data,
                timeout=30.0
            )
            response.raise_for_status()

            if response.status_code == 201:
                logger.info("Memory item successfully stored.")
                response_data = response.json()
                logger.info(f"Parse Server response data: {json.dumps(response_data, indent=2)}")
                logger.info(f"memoryChunkIds in Parse response: {response_data.get('memoryChunkIds')}")
                
                # Get user data from Parse response or create minimal user object
                user_data = response_data.get("user", {})
                user = {
                    "objectId": user_id,
                    "__type": "Pointer",  
                    "className": "_User", 
                    "displayName": user_data.get("displayName"),
                    "fullname": user_data.get("fullname"),
                    "profileimage": user_data.get("profileimage"),
                    "title": user_data.get("title"),
                    "isOnline": user_data.get("isOnline")
                }

                # Log the values we'll use for memoryChunkIds
                logger.info(f"memory_chunk_ids from local: {memory_chunk_ids}")
                logger.info(f"memoryChunkIds from response: {response_data.get('memoryChunkIds')}")
                logger.info(f"Will use: {response_data.get('memoryChunkIds', memory_chunk_ids)}")

                # Create base arguments dictionary for ParseStoredMemory
                stored_memory_args = {
                    "objectId": response_data["objectId"],
                    "createdAt": response_data.get("createdAt"),
                    "updatedAt": response_data.get("updatedAt"),
                    "ACL": parse_acl,
                    "content": memory_item.content,
                    "metadata": json.dumps(metadata),
                    "sourceType": metadata.get('sourceType', ""),
                    "context": context,
                    "title": metadata.get("title"),
                    "location": metadata.get("location"),
                    # Use data from the validated MemoryParseServer object
                    "emojiTags": data.get("emojiTags", []),
                    "emotionTags": data.get("emotionTags", []),
                    "hierarchicalStructures": data.get("hierarchicalStructures", ""),
                    "type": memory_item.type,
                    "sourceUrl": data.get("sourceUrl", ""),
                    "conversationId": data.get("conversationId", ""),
                    "memoryId": memory_id_str,
                    "topics": data.get("topics", []),
                    "steps": data.get("steps", []),
                    "current_step": data.get("current_step"),
                    # Use memoryChunkIds from Parse response instead of local variable
                    "memoryChunkIds": response_data.get("memoryChunkIds", memory_chunk_ids),
                    "user": ParseUserPointer(**user),
                    # Add optional pointers if they exist in data
                    "workspace": data.get("workspace"),
                    "post": data.get("post"),
                    "postMessage": data.get("postMessage")
                }

                # Add document-specific fields if type is DocumentMemoryItem
                if memory_item.type == "DocumentMemoryItem":
                    document_fields = {
                        "page_number": metadata.get('page_number'),
                        "total_pages": metadata.get('total_pages'),
                        "upload_id": metadata.get('upload_id'),
                        "filename": metadata.get('filename'),
                        "page": metadata.get('page'),
                        "file_url": metadata.get('file_url')
                    }
                    stored_memory_args.update(document_fields)

                # Create ParseStoredMemory object with all fields
                stored_memory = ParseStoredMemory(**stored_memory_args)
                
                logger.info(f"Created ParseStoredMemory with memoryChunkIds: {stored_memory.memoryChunkIds}")
                logger.info("Successfully created ParseStoredMemory object")
                return stored_memory
            else:
                logger.error(f"Failed to store memory item: {response.text}")
                return None

    except ValidationError as ve:
        logger.error(f"Validation error creating MemoryParseServer object: {ve}")
        return None
    except httpx.HTTPError as e:
        logger.error(f"HTTP error storing memory item: {e}")
        return None
    except Exception as e:
        logger.error(f"Error storing memory item: {e}")
        return None

def convert_neo_item_to_memory_item(neo_item: dict) -> dict:
    """
    Converts a Neo4j memory item dictionary into a MemoryItem dictionary with metadata.

    Args:
        neo_item (dict): The memory item fetched from Neo4j.

    Returns:
        MemoryItem: A MemoryItem dictionary with all required fields, including metadata.
    """
    # Extract metadata fields from neo_item, providing defaults if necessary
    metadata = {
        'workspace_id': neo_item.get('workspace_id', ''),
        'title': neo_item.get('title', 'Untitled'),
        'location': neo_item.get('location', 'online'),
        'hierarchical_structures': neo_item.get('hierarchical_structures', 'general'),
        'sourceType': neo_item.get('sourceType', 'papr'),
        'sourceUrl': neo_item.get('sourceUrl', ''),
        'current_step': neo_item.get('current_step', ''),
        'steps': neo_item.get('steps', []),
        'emoji_tags': neo_item.get('emoji_tags', ''),
        'topics': neo_item.get('topics', ''),
        'type': neo_item.get('type', 'text'),  # Add type field with default 'text'
        'memoryChunkIds': neo_item.get('memoryChunkIds', [])

        # Add other necessary fields as required
    }

    # Handle ACL-related fields if they exist
    acl_fields = {
        'user_read_access': neo_item.get('user_read_access', [neo_item.get('user_id')]),
        'user_write_access': neo_item.get('user_write_access', [neo_item.get('user_id')]),
        'workspace_read_access': neo_item.get('workspace_read_access', []),
        'workspace_write_access': neo_item.get('workspace_write_access', []),
        'role_read_access': neo_item.get('role_read_access', []),
        'role_write_access': neo_item.get('role_write_access', []),
    }

    # Incorporate ACL fields into metadata
    metadata.update(acl_fields)

    # Assign the constructed metadata to the neo_item
    neo_item['metadata'] = metadata

    # Optionally, remove fields from neo_item that are now part of metadata to avoid redundancy
    fields_to_remove = [
        'workspace_id',
        'title',
        'location',
        'hierarchical structures',
        'sourceType',
        'sourceUrl',
        'current_step',
        'steps',
        'emoji_tags',
        'topics',
        'user_read_access',
        'user_write_access',
        'workspace_read_access',
        'workspace_write_access',
        'role_read_access',
        'role_write_access',
    ]

    for field in fields_to_remove:
        neo_item.pop(field, None)

    return neo_item

def flatten_neo_item_to_parse_item(neo_item: dict) -> dict:
    """
    Flattens a Neo4j memory item by merging metadata into the top level and adjusting field formats.

    Args:
        neo_item (dict): The memory item fetched from Neo4j, potentially containing a 'metadata' dict.

    Returns:
        dict: A flattened memory item dictionary formatted for Parse Server.
    """
    if not isinstance(neo_item, dict):
        raise ValueError("Input neo_item must be a dictionary.")

    # Create a copy to avoid mutating the original neo_item
    parse_item = neo_item.copy()

    # Extract and remove 'metadata' from the neo_item
    metadata = parse_item.pop('metadata', {})

    if not isinstance(metadata, dict):
        logger.warning("Metadata is not a dictionary. Skipping flattening metadata.")
        metadata = {}

    # Define key mappings: Neo4j keys to Parse Server keys
    key_mappings = {
        'emoji_tags': 'emojiTags',
        'hierarchical_structures': 'hierarchicalStructures',
        'id': 'memoryId',
        # Add more mappings as needed
    }

    # Define fields to remove from the parse_item
    fields_to_remove = [
        'workspace_write_access',
        'workspace_read_access',
        'user_write_access',
        'user_read_access',
        'hierarchicalStructures',
        'role_read_access',
        'role_write_access'
    ]

    # Flatten metadata into parse_item with key renaming and data type conversions
    for key, value in metadata.items():
        parse_key = key_mappings.get(key, key)
        
        # Handle specific fields
        if parse_key == 'emojiTags':
            parse_item[parse_key] = [tag.strip() for tag in value.split(',') if tag.strip()] if isinstance(value, str) else (value if isinstance(value, list) else [])
        elif parse_key == 'hierarchicalStructures':
            # Rename and keep as string without converting to list
            parse_item[parse_key] = value.strip() if isinstance(value, str) else ""
        else:
            parse_item[parse_key] = value

    # Remove unnecessary fields after mapping
    for field in fields_to_remove:
        parse_item.pop(field, None)  # Use pop with default to avoid KeyError if field is not present

    return parse_item

def store_issue_memory_item(user_id: str, session_token: str, memory_item: MemoryItem):
    url = f"{PARSE_SERVER_URL}/parse/classes/Issue"
    logger.info(f"URL: {url}")

    memory_id_str = str(memory_item.id)
    metadata = str(memory_item.metadata)
    sourceType = memory_item.metadata.get('sourceType')
    sourceUrl = memory_item.metadata.get('sourceUrl')

    # Get memoryChunkIds from metadata if available
    memoryChunkIds = memory_item.metadata.get('memoryChunkIds', [])

    acl = {
        user_id: {
            "read": True,
            "write": True
        },
        "*": {
            "read": False,
            "write": False
        }
    }

    data = {
        "ACL": acl,
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
        "content": memory_item.content,
        "metadata": metadata,
        "sourceType": sourceType,
        "sourceUrl": sourceUrl, 
        "workspace": {
            "__type": "Pointer",
            "className": "WorkSpace",
            "objectId": memory_item.metadata.get("workspace_id")
        },
        "title": memory_item.metadata.get("title"),
        "url": memory_item.metadata.get("url"),
        "updatedAt": convert_to_parse_date(memory_item.metadata.get("updatedAt")),
        "creator": memory_item.metadata.get("creator"),
        "assignee": memory_item.metadata.get("assignee"),
        "project": memory_item.metadata.get("project"),
        "team": memory_item.metadata.get("team"),
        "tenant": memory_item.metadata.get("tenant"),
        "stream": memory_item.metadata.get("stream"),
        "status": memory_item.metadata.get("status"),
        "hierarchicalStructures": memory_item.metadata.get("hierarchical structures"),
        "type": memory_item.type,
        "linearId": memory_item.metadata.get("linear-id"),
        "connector": memory_item.metadata.get("connector"),
        "memoryId": memory_id_str,
        "topics": memory_item.metadata.get("topics"),
        "memoryChunkIds": memoryChunkIds  # Add pinecone chunk IDs
    }

    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_REST_API_KEY,   
        "Content-Type": "application/json"
    }

    data = {k: v for k, v in data.items() if v is not None}
    response = requests.post(url, headers=HEADERS, data=json.dumps(data))

    if response.status_code != 201:
        logger.error(f"Failed to store issue memory item: {response.text}")
        return None

    logger.info("Issue memory item successfully stored.")
    return response.json()

def convert_to_parse_date(date_string: str) -> dict:
    try:
        dt = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        return {
            "__type": "Date",
            "iso": dt.isoformat()
        }
    except ValueError:
        logger.error(f"Invalid date format: {date_string}")
        return None
    
def map_metadata_to_parse_fields(metadata: dict, node_name: str = None, relationships_json: dict = None) -> dict:
       """
       Maps specific metadata fields to Parse Server's expected fields.

       Parameters:
           metadata (dict): The original metadata dictionary.
           node_name (str): The name of the node.
           relationships_json (dict): The relationships in JSON format.

       Returns:
           dict: A dictionary containing only the fields relevant to Parse Server.
       """
       data = {}

       if 'title' in metadata:
           data["title"] = metadata["title"]

       if 'location' in metadata:
           data["location"] = metadata["location"]

       if 'emoji tags' in metadata:
           emoji_tags = metadata.get('emoji tags', [])
           if isinstance(emoji_tags, str):
               # Split the string by commas and strip whitespace
               emoji_tags = [tag.strip() for tag in emoji_tags.split(',')]
           data["emojiTags"] = emoji_tags

       if 'hierarchical structures' in metadata:
           data["hierarchicalStructures"] = metadata.get("hierarchical structures")

       if 'type' in metadata:
           data["type"] = metadata.get("type")

       if 'sourceUrl' in metadata:
           data["sourceUrl"] = metadata.get("sourceUrl")

       if 'conversationId' in metadata:
           data["conversationId"] = metadata.get("conversationId")

       if 'topics' in metadata:
           topics = metadata.get('topics', [])
           if isinstance(topics, str):
               # Split the string by commas and strip whitespace
               topics = [topic.strip() for topic in topics.split(',')]
           data["topics"] = topics

       if 'memoryChunkIds' in metadata:
           memory_chunk_ids = metadata.get('memoryChunkIds', [])
           if isinstance(memory_chunk_ids, str):
               try:
                   memory_chunk_ids = json.loads(memory_chunk_ids)
               except json.JSONDecodeError:
                   memory_chunk_ids = [id.strip() for id in memory_chunk_ids.strip('[]').split(',') if id.strip()]
           data["memoryChunkIds"] = memory_chunk_ids

       # Add other fields if necessary
       if node_name:
           data["node_name"] = node_name

       if relationships_json:
           data["relationship_json"] = relationships_json

       return data

async def delete_memory_item_parse(
    memory_item_id: str
) -> bool:
    """
    Asynchronously deletes a memory item from Parse Server.
    
    Args:
        memory_item_id (str): The objectId of the memory item to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    if not memory_item_id:
        logger.error("Memory item ID is required for deletion.")
        return False

    url = f"{PARSE_SERVER_URL}/parse/classes/Memory/{memory_item_id}"
    logger.info(f"Attempting to delete memory item with ID {memory_item_id}")
    logger.info(f"Delete URL: {url}")

    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }

    try:
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                url, 
                headers=headers, 
                timeout=30.0
            )
            
            # Log the response for debugging
            logger.info(f"Delete response status code: {response.status_code}")
            if response.text:  # Some DELETE responses might be empty
                logger.info(f"Delete response: {response.text}")

            if response.status_code == 200:
                logger.info(f"Successfully deleted memory item with ID: {memory_item_id}")
                return True
            else:
                logger.error(f"Failed to delete memory item. Status code: {response.status_code}")
                logger.error(f"Error response: {response.text}")
                return False

    except httpx.TimeoutException:
        logger.error(f"Timeout while deleting memory item: {memory_item_id}")
        return False
    except httpx.HTTPError as e:
        logger.error(f"HTTP error while deleting memory item: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while deleting memory item: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return False

def batch_store_memories(session_token: str, memory_items: list):
    url = f"{PARSE_SERVER_URL}/parse/batch"
    logger.info(f"Batch URL: {url}")

    requests_data = []

    for memory_item in memory_items:
        # Assuming 'topics' and 'emoji tags' are stored as lists in memory_item.metadata
        topics = memory_item.metadata.get('topics')
        emoji_tags = memory_item.metadata.get('emoji tags')

        # Convert UUID to string
        memory_id_str = str(memory_item.id)
        context = json.dumps(memory_item.context)  # Ensure context is a JSON string

        # Prepare the individual request data
        data = {
            "method": "POST",
            "path": "/parse/classes/Memory",
            "body": {
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": memory_item.metadata.get("user_id")
                },
                "content": memory_item.content,
                "context": context,
                "title": memory_item.metadata.get("title"),
                "location": memory_item.metadata.get("location"),
                "emojiTags": emoji_tags if isinstance(emoji_tags, list) else [],
                "hierarchicalStructures": memory_item.metadata.get("hierarchical structures"),
                "type": memory_item.metadata.get("type"),
                "sourceUrl": memory_item.metadata.get("sourceUrl"),
                "conversationId": memory_item.metadata.get("conversationId"),
                "memoryId": memory_id_str,
                "topics": topics if isinstance(topics, list) else [],
                # Add other fields if necessary
            }
        }

        # Remove any None values
        data["body"] = {k: v for k, v in data["body"].items() if v is not None}

        requests_data.append(data)

    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }

    # The batch request payload
    batch_payload = {
        "requests": requests_data
    }

    response = requests.post(url, headers=HEADERS, data=json.dumps(batch_payload))

    if response.status_code != 200:
        logger.error(f"Failed to batch store memory items: {response.text}")
        return None

    logger.info("Memory items successfully stored in batch.")
    return response.json()

async def retrieve_memory_item_by_pinecone_id(session_token: str, pinecone_id: str) -> Optional[Dict[str, Any]]:
    """
    Asynchronously retrieve a memory item from Parse Server by its Pinecone ID.
    
    Args:
        session_token (str): The session token for authentication
        pinecone_id (str): The Pinecone ID to search for
        
    Returns:
        Optional[Dict[str, Any]]: The memory item if found, None otherwise
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Memory"

    HEADERS = {
        "X-Parse-Application-Id": env.get("PARSE_APPLICATION_ID"),
        "X-Parse-REST-API-Key": env.get("PARSE_REST_API_KEY"),
        "X-Parse-Master-Key": env.get("PARSE_MASTER_KEY"),
        "Content-Type": "application/json"
    }

    # Strip chunk suffix (_0, _1, etc) to get base memory ID
    base_memory_id = pinecone_id.split('_')[0]

    # Create a query to find the memory item with either matching memoryId or in memoryChunkIds
    query = {
        "$or": [
            {"memoryId": base_memory_id},  # Match exact ID
            {"memoryId": pinecone_id},     # Match chunk ID
            {"memoryChunkIds": base_memory_id},  # Match as chunk
            {"memoryChunkIds": pinecone_id}      # Match as chunk
        ]
    }

    params = {
        "where": json.dumps(query),
        "limit": 1
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=30.0
            )
            
            response.raise_for_status()

            if response.status_code != 200:
                logger.error(f"Failed to retrieve memory item by Pinecone ID: {response.text}")
                return None

            results = response.json().get('results', [])
            
            if not results:
                logger.warning(f"No memory item found with Pinecone ID: {pinecone_id}")
                return None

            logger.info("Memory item successfully retrieved by Pinecone ID.")
            return results[0]  # Return the first (and should be only) result

    except httpx.TimeoutException:
        logger.error(f"Timeout while retrieving memory item for Pinecone ID: {pinecone_id}")
        return None
    except httpx.HTTPError as e:
        logger.error(f"HTTP error while retrieving memory item: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while retrieving memory item: {str(e)}")
        return None

async def retrieve_memory_item_parse(
    session_token: str, 
    memory_item_id: Optional[str] = None, 
    memory_chunk_ids: Optional[List[str]] = None
) -> Optional[ParseStoredMemory]:
    """
    Asynchronously retrieves a memory item from Parse Server.
    
    Args:
        session_token (str): The session token for authentication
        memory_item_id (Optional[str]): The memory ID to retrieve
        memory_chunk_ids (Optional[List[str]]): List of memory chunk IDs
        
    Returns:
        Optional[ParseStoredMemory]: The retrieved memory item
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Memory"
    
    query = {}
    if memory_item_id:
        # Strip chunk suffix (_0, _1, etc) to get base memory ID
        base_memory_id = memory_item_id.split('_')[0]
        query["$or"] = [
            {"memoryId": base_memory_id},  # Match exact ID
            {"memoryId": memory_item_id},  # Match chunk ID
            {"memoryChunkIds": base_memory_id},  # Match as chunk
            {"memoryChunkIds": memory_item_id}  # Match as chunk
        ]
    if memory_chunk_ids:
        # Strip chunk suffixes for all chunk IDs
        base_chunk_ids = [chunk_id.split('_')[0] for chunk_id in memory_chunk_ids]
        all_ids = memory_chunk_ids + base_chunk_ids
        query["$or"] = query.get("$or", []) + [
            {"memoryChunkIds": {"$in": all_ids}},
            {"memoryId": {"$in": all_ids}}
        ]
    
    logger.debug(f"Query for Parse Server: {query}")
    
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    params = {
        "where": json.dumps(query)
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, 
                headers=headers, 
                params=params,
                timeout=30.0
            )
            
            response.raise_for_status()
            results = response.json().get('results', [])
            
            if not results:
                logger.warning(f"No memory item found with the given criteria.")
                return None
                
            memory_data = results[0]
            
            # Convert to ParseStoredMemory
            try:
                # Extract metadata
                metadata = memory_data.get('metadata', '{}')
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                
                # Get user data
                user_data = memory_data.get("user", {})
                user = {
                    "objectId": user_data.get("objectId"),
                    "displayName": user_data.get("displayName"),
                    "fullname": user_data.get("fullname"),
                    "profileimage": user_data.get("profileimage"),
                    "title": user_data.get("title"),
                    "isOnline": user_data.get("isOnline")
                }
                
                stored_memory = ParseStoredMemory(
                    objectId=memory_data["objectId"],
                    createdAt=memory_data.get("createdAt"),
                    updatedAt=memory_data.get("updatedAt"),
                    ACL=memory_data.get("ACL", {}),
                    content=memory_data.get("content", ""),
                    metadata=json.dumps(metadata),
                    sourceType=memory_data.get("sourceType", ""),
                    context=memory_data.get("context", ""),
                    title=memory_data.get("title"),
                    location=memory_data.get("location"),
                    emojiTags=memory_data.get("emojiTags", []),
                    emotionTags=memory_data.get("emotionTags", []),
                    hierarchicalStructures=memory_data.get("hierarchicalStructures", ""),
                    type=memory_data.get("type", ""),
                    sourceUrl=memory_data.get("sourceUrl", ""),
                    conversationId=memory_data.get("conversationId", ""),
                    memoryId=memory_data.get("memoryId", ""),
                    topics=memory_data.get("topics", []),
                    steps=memory_data.get("steps", []),
                    current_step=memory_data.get("current_step"),
                    memoryChunkIds=memory_data.get("memoryChunkIds", []),
                    user=ParseUserPointer(**user),
                    workspace=memory_data.get("workspace"),
                    post=memory_data.get("post"),
                    postMessage=memory_data.get("postMessage")
                )
                
                return stored_memory
                
            except ValidationError as ve:
                logger.error(f"Validation error creating ParseStoredMemory: {ve}")
                return None
                
    except httpx.HTTPError as e:
        logger.error(f"HTTP error retrieving memory item: {e}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving memory item: {e}")
        return None

def retrieve_memory_item_with_user(session_token: str, memory_item_id: str):
    """
    Retrieves a memory item from Parse Server by its memoryId, including the associated user object.

    Parameters:
        session_token (str): The session token of the authenticated user.
        memory_item_id (str): The memoryId of the memory item to retrieve.

    Returns:
        dict or None: The memory item data with the user object if found, else None.
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Memory"
    
    query = json.dumps({"memoryId": memory_item_id})
    
    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Specify the fields to include, using dot notation for nested user fields
    keys = (
        "ACL,"
        "user.objectId,user.displayName,user.fullname,user.profileimage,user.title,user.isOnline,"
        "content,context,topics,emojiTags,hierarchicalStructures,type,sourceUrl,sourceType,title,memoryId,steps,current_step,createdAt,updatedAt"
    )
    
    params = {
        "where": query,
        "keys": keys  # Specify fields to include
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        logger.error(f"Failed to retrieve memory item with user: {response.text}")
        return None
    
    results = response.json().get('results', [])

    # Log the raw response for debugging
    logger.debug(f"Raw Parse Server Response single: {response.json()}")
    
    if not results:
        logger.warning(f"No memory item found with memoryId: {memory_item_id}")
        return None
    
    logger.info("Memory item with user successfully retrieved.")
    memory_item = results[0]
    
    # Define the desired fields for the user object
    desired_user_fields = [
        'objectId',
        'displayName',
        'fullname',
        'profileimage',
        'title',
        'isOnline'
    ]
    
    # Clean up the user object by keeping only the desired fields
    if 'user' in memory_item:
        user = memory_item['user']
        # Create a new user dictionary with only the desired fields
        filtered_user = {k: v for k, v in user.items() if k in desired_user_fields}
        memory_item['user'] = filtered_user
    
    return memory_item  # Return the first matching memory item with filtered user

def retrieve_memory_items_with_users(session_token: str, memory_item_ids: list, chunk_base_ids: list, class_name: str):
    """
    Retrieves multiple memory items from Parse Server by their memoryIds, including the associated user objects.

    Parameters:
        session_token (str): The session token of the authenticated user.
        memory_item_ids (list): A list of memoryIds of the memory items to retrieve.
        chunk_base_ids (list): List of base memory IDs without chunk numbers
        class_name (str): The class name in Parse Server to query.

    Returns:
        dict: A dictionary containing:
            - 'results': List of memory items with user objects
            - 'missing_memory_ids': List of memory IDs that weren't found
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/{class_name}"
    
    # Log input parameters for debugging
    logger.info(f"Retrieving memory items with users. Memory IDs count: {len(memory_item_ids)}")
    logger.info(f"First few memory IDs: {memory_item_ids[:5] if memory_item_ids else []}")
    
    # Formulate the query
    query = {
        "$or": [
            {"memoryId": {"$in": memory_item_ids}},
            {"memoryChunkIds": {"$in": memory_item_ids}},
            {"memoryId": {"$in": chunk_base_ids}}
        ]
    }
    
    # Specify the fields to include
    keys = (
        "objectId,createdAt,updatedAt,ACL,content,metadata,sourceType,context,title,location,"
        "emojiTags,hierarchicalStructures,type,sourceUrl,conversationId,memoryId,topics,steps,"
        "current_step,memoryChunkIds,user.objectId,user.displayName,user.fullname,user.profileimage,"
        "user.title,user.isOnline,workspace,post,postMessage,matchingChunkIds"
    )

    params = {
        "where": json.dumps(query),
        "keys": keys  # Specify fields to include
    }

    HEADERS = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Session-Token": session_token,
        "Content-Type": "application/json"
    }
    
    # Log the query for debugging
    logger.debug(f"Query parameters: {params}")
    
    response = requests.get(url, headers=HEADERS, params=params)

    # Log response status and details
    logger.info(f"Response status code: {response.status_code}")
    
    if response.status_code != 200:
        logger.error(f"Failed to retrieve memory items with users: {response.text}")
        return []
    
    results = response.json().get('results', [])
    logger.info(f"Retrieved {len(results)} memory items from Parse Server")
    
    # Log the memory IDs that were found vs not found
    found_memory_ids = [item.get('memoryId') for item in results]
    missing_memory_ids = list(set(memory_item_ids) - set(found_memory_ids))
    
    if missing_memory_ids:
        logger.warning(f"Missing memory items: {len(missing_memory_ids)}")
        logger.warning(f"First few missing memory IDs: {missing_memory_ids[:5]}")
    
    # Define the desired fields for the user object
    desired_user_fields = [
        'objectId',
        'displayName',
        'fullname',
        'profileimage',
        'title',
        'isOnline'
    ]
    
    # Clean up each memory item's user object
    for memory_item in results:
        if 'user' in memory_item:
            user = memory_item['user']
            # Create a new user dictionary with only the desired fields
            filtered_user = {k: v for k, v in user.items() if k in desired_user_fields}
            memory_item['user'] = filtered_user
    
    return {
        'results': results,
        'missing_memory_ids': missing_memory_ids
    }

def retrieve_memory_items(session_token: str, memory_item_ids: list, class_name: str):
    url = f"{PARSE_SERVER_URL}/parse/classes/{class_name}"
    
    # Formulating the query to match any memory item whose objectId is in memory_item_ids
    params = {
        "where": json.dumps({
            "memoryId": {
                "$in": memory_item_ids
            }
        })
    }

    HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
    }
    
    
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        logger.error(f"Failed to retrieve memory items: {response.text}")
        return None
    
    logger.info("Memory items successfully retrieved.")
    return response.json().get('results', [])

def retrieve_multiple_memory_items(session_token: str, memory_item_ids: list, chunk_base_ids: list):
    """
    Retrieves multiple memory items with their associated users, deduplicating by memoryId.
    
    Parameters:
        session_token (str): The session token of the authenticated user
        memory_item_ids (list): List of memory IDs to retrieve
        chunk_base_ids (list): List of base memory IDs
        
    Returns:
        dict: A dictionary containing:
            - 'results': List of retrieved memory items with user data (deduplicated by memoryId)
            - 'missing_memory_ids': List of memory IDs that weren't found
    """
    memory_class = "Memory"
    try:
        response = retrieve_memory_items_with_users(session_token, memory_item_ids, chunk_base_ids, memory_class)
        
        # Deduplicate results based on memoryId
        seen_memory_ids = set()
        deduplicated_results = []
        
        for item in response["results"]:
            memory_id = item.get('memoryId')
            if memory_id and memory_id not in seen_memory_ids:
                seen_memory_ids.add(memory_id)
                deduplicated_results.append(item)
        
        # Update response with deduplicated results
        response["results"] = deduplicated_results
        
        logger.info(f'Retrieved {len(response["results"])} deduplicated items from Parse Server.')
        logger.info(f'Missing {len(response["missing_memory_ids"])} items.')
        if response["missing_memory_ids"]:
            logger.info(f'First few missing IDs: {response["missing_memory_ids"][:5]}')
        return response
    except Exception as exc:
        logger.error(f"Error retrieving Memory items: {exc}")
        return {'results': [], 'missing_memory_ids': memory_item_ids}

async def retrieve_memory_items_with_users_async(
    session_token: str, 
    memory_item_ids: list, 
    chunk_base_ids: list, 
    class_name: str = "Memory"
)  -> MemoryRetrievalResult:
    """
    Async version: Retrieves multiple memory items from Parse Server by their memoryIds.

    Parameters:
        session_token (str): The session token of the authenticated user.
        memory_item_ids (List[str]): A list of memoryIds of the memory items to retrieve.
        chunk_base_ids (List[str]): List of base memory IDs without chunk numbers
        class_name (str): The class name in Parse Server to query.

    Returns:
        MemoryRetrievalResult: A dictionary containing:
            - 'results': List[ParseStoredMemory] - List of memory items with user objects
            - 'missing_memory_ids': List[str] - List of memory IDs that weren't found
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/{class_name}"
    
    # Log input parameters for debugging
    logger.info(f"Retrieving memory items with users. Memory IDs count: {len(memory_item_ids)}")
    logger.info(f"First few memory IDs: {memory_item_ids[:5] if memory_item_ids else []}")
    
    # Deduplicate IDs
    unique_memory_ids = list(set(memory_item_ids))
    unique_chunk_base_ids = list(set(chunk_base_ids))

    # For memoryId search, combine and dedupe both sets
    all_ids = list(set(unique_memory_ids + unique_chunk_base_ids))
    
    logger.info(f"Deduplicated memory IDs count: {len(unique_memory_ids)}")
    logger.info(f"Deduplicated chunk base IDs count: {len(unique_chunk_base_ids)}")
    logger.info(f"Combined unique IDs count: {len(all_ids)}")

    # Constants for batch size
    MAX_IDS_PER_BATCH = 30
    max_retries = 3
    
    # Specify the fields to include
    keys = (
        "objectId,createdAt,updatedAt,ACL,content,metadata,sourceType,context,title,location,"
        "emojiTags,hierarchicalStructures,type,sourceUrl,conversationId,memoryId,topics,steps,"
        "current_step,memoryChunkIds,user.objectId,user.displayName,user.fullname,user.profileimage,"
        "user.title,user.isOnline,workspace,post,postMessage,matchingChunkIds"
    )

    async def execute_batch_query(all_ids: List[str], memory_ids: List[str] = None) -> List[Dict]:
        """Execute a single batch query"""
        # Build query for this batch
        query = {
            "$or": [
                {"memoryId": {"$in": all_ids}},  # Search memoryId in base IDs
            ]
        }
        
        # Only add memoryChunkIds search if chunk_ids is provided
        if memory_ids:
            query["$or"].append({"memoryChunkIds": {"$in": memory_ids}})  # Search chunks in memory IDs

        params = {
            "where": json.dumps(query),
            "keys": keys
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
                "X-Parse-Session-Token": session_token,
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate"
            }
        ) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(url, params=params, follow_redirects=True)
                    response.raise_for_status()
                    return response.json().get('results', [])
                except httpx.HTTPError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Batch query failed after all retries: {str(e)}")
                        raise
                    await asyncio.sleep(1 * (attempt + 1))

    async def process_batches() -> List[Dict]:
        """Process all IDs in batches"""
        
        # Split into batches
        all_ids_batches = [all_ids[i:i + MAX_IDS_PER_BATCH] 
                       for i in range(0, len(all_ids), MAX_IDS_PER_BATCH)]
        memory_ids_batches = [unique_memory_ids[i:i + MAX_IDS_PER_BATCH] 
                         for i in range(0, len(unique_memory_ids), MAX_IDS_PER_BATCH)]
        
        # Create tasks for all batches
        tasks = []
        for i, all_ids_batch in enumerate(all_ids_batches):
            # Find corresponding memory batch for chunk search
            memory_ids_batch = memory_ids_batches[i] if i < len(memory_ids_batches) else None
            tasks.append(execute_batch_query(all_ids_batch, memory_ids_batch))

        # Execute all batches in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and handle any exceptions
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Failed to process batch: {str(result)}")
                continue
            all_results.extend(result)
            
        return all_results

    try:
        # Process all batches
        raw_results = await process_batches()
        
        # Convert raw results to ParseStoredMemory objects
        results = []
        for item in raw_results:
            try:
                # Clean up user data first
                if 'user' in item:
                    user_data = item['user']
                    item['user'] = {
                        'objectId': user_data.get('objectId'),
                        'displayName': user_data.get('displayName'),
                        'fullname': user_data.get('fullname'),
                        'profileimage': user_data.get('profileimage', {}).get('url') if isinstance(user_data.get('profileimage'), dict) else user_data.get('profileimage'),
                        'title': user_data.get('title'),
                        'isOnline': user_data.get('isOnline')
                    }

                # Ensure required fields exist with proper types
                cleaned_item = {
                    'objectId': item.get('objectId'),
                    'createdAt': item.get('createdAt'),
                    'updatedAt': item.get('updatedAt'),
                    'ACL': item.get('ACL', {}),
                    'content': item.get('content', ''),
                    'metadata': item.get('metadata', '{}'),
                    'sourceType': item.get('sourceType', ''),
                    'context': item.get('context', ''),
                    'title': item.get('title'),
                    'location': item.get('location'),
                    'emojiTags': item.get('emojiTags', []),
                    'hierarchicalStructures': item.get('hierarchicalStructures', ''),
                    'type': item.get('type', 'TextMemoryItem'),
                    'sourceUrl': item.get('sourceUrl', ''),
                    'conversationId': item.get('conversationId', ''),
                    'memoryId': item.get('memoryId'),
                    'topics': item.get('topics', []),
                    'steps': item.get('steps', []),
                    'current_step': item.get('current_step'),
                    'memoryChunkIds': item.get('memoryChunkIds', []),
                    'user': item.get('user'),
                }

                # Add optional pointers only if they exist
                if 'workspace' in item and isinstance(item['workspace'], dict):
                    cleaned_item['workspace'] = {
                        '__type': 'Pointer',
                        'className': 'WorkSpace',
                        'objectId': item['workspace'].get('objectId')
                    }

                if 'post' in item and isinstance(item['post'], dict):
                    cleaned_item['post'] = {
                        '__type': 'Pointer',
                        'className': 'Post',
                        'objectId': item['post'].get('objectId')
                    }

                # Remove any None values
                cleaned_item = {k: v for k, v in cleaned_item.items() if v is not None}

                # Validate the cleaned item
                memory = ParseStoredMemory.model_validate(cleaned_item)
                logger.debug(f"Successfully validated memory {memory.memoryId} as ParseStoredMemory")
                results.append(memory)
            except Exception as e:
                logger.error(f"Failed to validate memory item as ParseStoredMemory: {e}")
                logger.error(f"Problematic item: {item}")
                continue

        # Process found vs missing memory IDs
        found_memory_ids = [item.memoryId for item in results]
        missing_memory_ids = list(set(memory_item_ids) - set(found_memory_ids))

        if missing_memory_ids:
            logger.warning(f"Missing memory items: {len(missing_memory_ids)}")
            logger.warning(f"First few missing memory IDs: {missing_memory_ids[:5]}")
        
        # Deduplicate results based on memoryId
        seen_memory_ids = set()
        deduplicated_results: List[ParseStoredMemory] = []
        
        for item in results:
            if item.memoryId and item.memoryId not in seen_memory_ids:
                seen_memory_ids.add(item.memoryId)
                deduplicated_results.append(item)
        
        # Process found vs missing memory IDs using deduplicated results
        found_memory_ids = [item.memoryId for item in deduplicated_results]
        missing_memory_ids = list(set(memory_item_ids) - set(found_memory_ids))
        
        logger.info(f'Retrieved {len(deduplicated_results)} deduplicated items from Parse Server.')
        logger.info(f'Missing {len(missing_memory_ids)} items.')
        if missing_memory_ids:
            logger.info(f'First few missing IDs: {missing_memory_ids[:5]}')
        
        return {
            'results': deduplicated_results,
            'missing_memory_ids': missing_memory_ids
        }

    except Exception as e:
        logger.error(f"Failed to retrieve memory items: {str(e)}")
        return {'results': [], 'missing_memory_ids': memory_item_ids}

async def store_connector_user_id(session_token: str, user_id: str, connector_type: str, connector_user_id: str):
    """
    Adds a connector user ID to the ConnectorUserId class in the Parse server.

    Parameters:
        session_token (str): The session token of the authenticated user.
        user_id (str): The objectId of the authenticated user.
        connector_type (str): The type/name of the connector (e.g., 'slack', 'github').
        connector_user_id (str): The user ID from the connector app.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    if not PARSE_SERVER_URL:
        logger.error("PARSE_SERVER_URL environment variable is not set.")
        return False

    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Session-Token": session_token,
        "Content-Type": "application/json"
    }

    # First, check if the ConnectorUserId already exists
    find_url = f"{PARSE_SERVER_URL}/parse/classes/ConnectorUserId"
    
    query = {
        "where": json.dumps({
            "connector_type": connector_type,
            "connector_user_id": connector_user_id
        })
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(find_url, headers=headers, params=query)
            response.raise_for_status()
            results = response.json().get("results", [])

            if results:
                logger.info(f"'ConnectorUserId' entry already exists for connector '{connector_type}' with user ID '{connector_user_id}'.")
                return False  # Already exists

            # If not exists, proceed to create
            create_url = find_url  # Same endpoint for POST
            data = {
                "connector_type": connector_type,
                "connector_user_id": connector_user_id,
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                }
            }

            create_response = await client.post(create_url, headers=headers, json=data)
            create_response.raise_for_status()

            logger.info(f"Successfully created 'ConnectorUserId' entry for connector '{connector_type}' with user ID '{connector_user_id}'.")
            return True

    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error occurred while handling 'ConnectorUserId': {http_err} - {response.text}")
    except Exception as err:
        logger.error(f"An unexpected error occurred: {err}")

    return False

def find_user_by_connector_id(session_token: str, connector_type: str, connector_user_id: str):
    """
    Retrieves a Papr _User object based on a single connector user ID for a specific connector.

    Parameters:
        connector_type (str): The type/name of the connector (e.g., 'slack', 'github').
        connector_user_id (str): The user ID from the connector app.

    Returns:
        dict or None: A dictionary containing the user's objectId and other public fields if found, else None.
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/ConnectorUserId"
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
         "X-Parse-Session-Token": session_token,
        "Content-Type": "application/json"
    }
    params = {
        "where": json.dumps({
            "connector_type": connector_type,
            "connector_user_id": connector_user_id
        }),
        "include": "user"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            user = results[0]["user"]
            return user  # Contains the user's objectId and other public fields
        return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while querying users: {http_err} - {response.text}")
    except Exception as err:
        logger.error(f"An unexpected error occurred: {err}")

    return None


async def find_user_by_connector_ids(
    session_token: str, 
    connector_type: str, 
    connector_user_ids: List[str]
) -> List[str]:
    """
    Async version: Retrieves Papr _User objectIds based on a list of connector user IDs.

    Args:
        session_token (str): The session token of the authenticated user
        connector_type (str): The type/name of the connector (e.g., 'slack', 'github')
        connector_user_ids (List[str]): A list of user IDs from the connector app

    Returns:
        List[str]: A list of Papr _User objectIds that match the provided connector user IDs
    """
    if not PARSE_SERVER_URL:
        logger.error("PARSE_SERVER_URL environment variable is not set.")
        return []

    url = f"{PARSE_SERVER_URL}/parse/classes/ConnectorUserId"

    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Session-Token": session_token,
        "Content-Type": "application/json"
    }
    logger.info(f"headers: {headers}")

    # Construct the 'where' clause using the '$in' operator for multiple IDs
    where_clause = {
        "connector_type": connector_type,
        "connector_user_id": {"$in": connector_user_ids}
    }
    logger.info(f"where_clause: {where_clause}")

    params = {
        "where": json.dumps(where_clause),
        "include": "user",
        "limit": 1000  # Adjust limit as needed (max 1000 in Parse Server)
    }

    logger.info(f"params: {params}")
    logger.info(f"url: {url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            # Extract the 'objectId' of the 'user' from each result
            object_ids = [result["user"]["objectId"] for result in results if "user" in result]
            logger.info(f"Retrieved {len(object_ids)} user objectIds for connector '{connector_type}'.")
            return object_ids
        
        except httpx.HTTPError as http_err:
            logger.error(f"HTTP error occurred while querying users: {http_err}")
            return []
        except Exception as err:
            logger.error(f"An unexpected error occurred: {err}")
            return []

async def get_workspaceId_using_tenantId(session_token: str, tenant_id: str) -> str:
    """
    Asynchronously retrieves the objectId of a WorkSpace based on the provided tenantId.

    Parameters:
        session_token (str): The session token of the authenticated user.
        tenant_id (str): The tenantId to query the WorkSpace.

    Returns:
        str or None: The objectId of the WorkSpace if found, else None.
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace"
    
    # Define the query parameters to filter by tenantId
    params = {
        "where": json.dumps({
            "tenantId": tenant_id
        }),
        "limit": 1  # Assuming tenantId is unique
    }
    
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Session-Token": session_token,
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Fetching workspace for tenant_id: {tenant_id}")
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            results = response.json().get('results', [])
            
            if not results:
                logger.warning(f"No WorkSpace found with tenantId: {tenant_id}")
                return None
            
            workspace = results[0]
            workspace_id = workspace.get('objectId')
            logger.info(f"WorkSpace successfully retrieved: {workspace_id}")
            return workspace_id
            
    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error occurred while retrieving workspace: {http_err} - {response.text}")
        return None
    except Exception as err:
        logger.error(f"Unexpected error occurred while retrieving workspace: {err}")
        return None

async def get_user_goals_async(user_id: str, session_token: str) -> List[dict]:
    """Async version of get_user_goals
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        
    Returns:
        List[dict]: A list of goal dictionaries. Returns empty list if request fails.
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Goal"
    params = {
        "where": json.dumps({
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        })
    }
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('results', [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to get user goals: {str(e)}")
            return []

async def get_user_usecases_async(user_id: str, session_token: str) -> List[dict]:
    """Async version of get_user_usecases
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        
    Returns:
        List[dict]: A list of usecase dictionaries containing name, description, and createdAt fields.
                   Returns empty list if request fails.
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Usecase"
    where_clause = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
    }
    params = {
        "where": json.dumps(where_clause),
        "keys": "name,description,createdAt",  # Moved outside where clause
        "order": "-createdAt",  # Moved outside where clause
        "limit": 10  # Moved outside where clause
    }
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('results', [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to get user usecases: {str(e)}")
            return []

async def get_user_memGraph_schema_async(user_id: str, session_token: str) -> dict:
    """Async version of get_user_memGraph_schema using httpx
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        
    Returns:
        dict or None: The memory graph schema if found, None otherwise
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraph"
    params = {
        "where": json.dumps({
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }),
        "include": "nodes,relationships"
    }
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('results'):
                return data['results'][0]
            return None
        except httpx.HTTPError as e:
            logger.error(f"Failed to get memory graph schema: {str(e)}")
            return None

async def create_usecase_async(user_id: str, session_token: str, name: str, description: str) -> dict:
    """Async version of create_usecase using httpx
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        name (str): The name of the usecase
        description (str): The description of the usecase
        
    Returns:
        dict or None: The created usecase data if successful, None otherwise
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Usecase"
    data = {
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
        "name": name,
        "description": description
    }
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to create usecase: {str(e)}")
            return None

async def add_list_of_usecases_async(user_id: str, session_token: str, use_cases: list):
    """Async version of add_list_of_usecases"""
    async def add_single_usecase(use_case):
        if not use_case.get('name') or not use_case.get('description'):
            logger.info(f"Invalid use case: {use_case}")
            return None
        response = await create_usecase_async(user_id, session_token, use_case['name'], use_case['description'])
        if response is not None:
            logger.info(f"Use case added successfully: {use_case['name']}")
        else:
            logger.info(f"Failed to add use case: {use_case['name']}")
        return response

    # Create tasks for all use cases
    tasks = [add_single_usecase(use_case) for use_case in use_cases]
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    return results

async def create_goal_async(
    user_id: str, 
    session_token: str, 
    title: str, 
    description: str = None, 
    key_results: List[dict] = None
) -> Optional[dict]:
    """Async version of create_goal using httpx
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        title (str): The title of the goal
        description (str, optional): The description of the goal. Defaults to None.
        key_results (List[dict], optional): List of key results for the goal. Defaults to None.
        
    Returns:
        Optional[dict]: The created goal data if successful, None otherwise
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Goal"
    data = {
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
        "title": title,
        "description": description or "",
        "keyResults": key_results or []
    }
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to create goal: {str(e)}")
            return None

async def add_list_of_goals_async(user_id: str, session_token: str, goals: List[dict]) -> List[Optional[dict]]:
    """Async version of add_list_of_goals that processes multiple goals concurrently
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        goals (List[dict]): List of goal dictionaries containing title, description, and key_results
        
    Returns:
        List[Optional[dict]]: List of created goal data, None for any goals that failed to create
    """
    async def add_single_goal(goal: dict) -> Optional[dict]:
        key_results = goal.get('key_results', [])
        description = goal.get('description', '')
        title = goal.get('title', '')
        
        response = await create_goal_async(user_id, session_token, title, description, key_results)
        if response is not None:
            logger.info(f"Goal added successfully: {title}")
        else:
            logger.error(f"Failed to add goal: {title}")
        return response

    # Create tasks for all goals
    tasks = [add_single_goal(goal) for goal in goals]
    
    # Wait for all tasks to complete concurrently
    results = await asyncio.gather(*tasks)
    
    return results

async def create_memory_graph_node_async(user_id: str, session_token: str, name: str) -> Optional[str]:
    """Async version of create_memory_graph_node using httpx
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        name (str): The name of the memory graph node
        
    Returns:
        Optional[str]: The objectId of the created/existing node if successful, None otherwise
    """
    query_url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraphNode"
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Check if node already exists
    params = {
        "where": json.dumps({
            "name": name,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        })
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Check if node exists
            response = await client.get(query_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                return data['results'][0]['objectId']
            
            # Create new node if it doesn't exist
            data = {
                "name": name,
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                }
            }
            
            create_response = await client.post(query_url, headers=headers, json=data)
            create_response.raise_for_status()
            result = create_response.json()
            return result['objectId']
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to create memory graph node: {str(e)}")
            return None

async def create_memory_graph_relationship_async(user_id: str, session_token: str, name: str) -> Optional[str]:
    """Async version of create_memory_graph_relationship using httpx
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        name (str): The name of the relationship
        
    Returns:
        Optional[str]: The objectId of the created/existing relationship if successful, None otherwise
    """
    query_url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraphRelationship"
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    # Check if relationship already exists
    params = {
        "where": json.dumps({
            "name": name,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        })
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Check if relationship exists
            response = await client.get(query_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                return data['results'][0]['objectId']
            
            # Create new relationship if it doesn't exist
            data = {
                "name": name,
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                }
            }
            
            create_response = await client.post(query_url, headers=headers, json=data)
            create_response.raise_for_status()
            result = create_response.json()
            return result['objectId']
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to create memory graph relationship: {str(e)}")
            return None

async def create_memory_graph_async(user_id: str, session_token: str, schema: dict) -> Optional[dict]:
    """Async version of create_memory_graph using httpx
    
    Args:
        user_id (str): The user's ID
        session_token (str): The session token for authentication
        schema (dict): The memory graph schema containing nodes and relationships
        
    Returns:
        Optional[dict]: The created memory graph data if successful, None otherwise
    """
    # Extract nodes and relationships from schema
    nodes = schema.get('nodes', [])
    relationships = schema.get('relationships', [])
    
    # Create nodes and relationships concurrently
    node_tasks = [create_memory_graph_node_async(user_id, session_token, node['name']) for node in nodes]
    relationship_tasks = [create_memory_graph_relationship_async(user_id, session_token, rel['name']) for rel in relationships]
    
    # Wait for all tasks to complete
    node_ids = await asyncio.gather(*node_tasks)
    relationship_ids = await asyncio.gather(*relationship_tasks)
    
    # Filter out None values
    node_ids = [nid for nid in node_ids if nid is not None]
    relationship_ids = [rid for rid in relationship_ids if rid is not None]
    
    # Create the memory graph
    url = f"{PARSE_SERVER_URL}/parse/classes/MemoryGraph"
    data = {
        "user": {
            "__type": "Pointer",
            "className": "_User",
            "objectId": user_id
        },
        "schema": schema,
        "nodes": [
            {
                "__type": "Pointer",
                "className": "MemoryGraphNode",
                "objectId": node_id
            } for node_id in node_ids
        ],
        "relationships": [
            {
                "__type": "Pointer",
                "className": "MemoryGraphRelationship",
                "objectId": relationship_id
            } for relationship_id in relationship_ids
        ]
    }
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Successfully created memory graph: {result['objectId']}")
            return result
        except httpx.HTTPError as e:
            logger.error(f"Failed to create memory graph: {str(e)}")
            return None

async def update_memory_item(
    session_token: str, 
    updated_memory_item: dict, 
    node_and_relationships: Optional[dict] = None
) -> UpdateMemoryResponse:
    try:
        # Get the memory ID and strip any chunk suffix
        # Handle both direct access and .get() patterns
        memory_objectId = updated_memory_item.get('objectId') or updated_memory_item['objectId']

        memory_id = updated_memory_item.get('id') or updated_memory_item.get('memoryId')

        # First get the existing memory to preserve ACL if needed
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token,
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            get_url = f"{PARSE_SERVER_URL}/parse/classes/Memory/{memory_objectId}"
            get_response = await client.get(get_url, headers=headers)
            
            if get_response.status_code != 200:
                return UpdateMemoryResponse.error_response(
                    f"Failed to get existing memory item: {get_response.text}",
                    code=get_response.status_code
                )
            
            existing_memory = get_response.json()
            existing_acl = existing_memory.get('ACL', {})
            logger.info(f"Existing ACL: {existing_acl}")
        
        if not memory_id:
            return UpdateMemoryResponse.error_response(
                "Memory ID is required",
                code=400
            )
            
        if not memory_objectId:
            return UpdateMemoryResponse.error_response(
                "Updated memory item does not have an objectId",
                code=400
            )
        
        # Strip chunk suffix (_0, _1, etc) to get base memory ID
        base_memory_id = memory_id.split('_')[0]
        updated_memory_item['memoryId'] = base_memory_id
        logger.info(f"Using base memory ID: {base_memory_id}")
        logger.info(f"memory_objectId: {memory_objectId}")
        
        if not memory_objectId:
            return UpdateMemoryResponse.error_response(
                "Updated memory item does not have an objectId",
                code=400
            )
        
        # Extract metrics if they exist
        metrics = updated_memory_item.get('metrics', {})
        operation_costs = metrics.get('operation_costs', {})

        # Extract metadata and convert string fields to lists
        metadata = updated_memory_item.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.error("Failed to parse metadata string")
                metadata = {}
        # Pre-process list fields to ensure proper format before Pydantic validation
        emoji_tags = convert_comma_string_to_list(
            metadata.get('emojiTags') or 
            metadata.get('emoji_tags') or 
            metadata.get('emoji tags')
        )
        emotion_tags = convert_comma_string_to_list(
            metadata.get('emotionTags') or 
            metadata.get('emotion_tags') or 
            metadata.get('emotion tags')
        )
        topics = convert_comma_string_to_list(metadata.get('topics'))
        memory_chunk_ids = metadata.get('memoryChunkIds') or updated_memory_item.get('memoryChunkIds', [])
        steps = convert_comma_string_to_list(metadata.get('steps'))

        # Update metadata with converted lists only if they exist
        if emoji_tags:
            metadata['emojiTags'] = emoji_tags
        if emotion_tags:
            metadata['emotionTags'] = emotion_tags
        if topics:
            metadata['topics'] = topics
        if memory_chunk_ids:
            metadata['memoryChunkIds'] = memory_chunk_ids
        if steps:
            metadata['steps'] = steps

        # Get ACL from either metadata or direct ACL field, fallback to existing ACL
        new_acl = None
        if 'ACL' in updated_memory_item:  # Direct ACL update
            new_acl = updated_memory_item['ACL']
            logger.info(f"New ACL: {new_acl}")
        elif metadata:  # ACL in metadata
            new_acl = convert_acl(metadata)
            logger.info(f"New ACL conver_acl: {new_acl}")
        # If no new ACL provided or empty ACL, use existing
        final_acl = new_acl if new_acl else existing_acl
        logger.info(f"Final ACL to use: {final_acl}")

        # Create base parse data with minimal fields first
        base_data = {
            "ACL": final_acl,
            "user": ParsePointer(
                __type="Pointer",
                className="_User",
                objectId=metadata.get("user_id")
            ),
            "memoryId": base_memory_id,
        }



        logger.info(f"base_data inside update_memory_item: {base_data}")

        # Create a MemoryParseServer instance with base data first
        parse_memory = MemoryParseServer(**base_data)

        # Add metrics fields if they exist
        if 'metrics' in updated_memory_item:
            metrics = updated_memory_item['metrics']
            metrics_mapping = {
                'totalProcessingCost': metrics.get('total_cost'),
                'tokenSize': metrics.get('token_size'),
                'storageSize': metrics.get('storage_size'),
                'usecaseGenerationCost': metrics.get('operation_costs', {}).get('usecase_generation'),
                'schemaGenerationCost': metrics.get('operation_costs', {}).get('schema_generation'),
                'relatedMemoriesCost': metrics.get('operation_costs', {}).get('related_memories'),
                'bigbirdEmbeddingCost': metrics.get('operation_costs', {}).get('bigbird_embedding'),
                'sentenceBertCost': metrics.get('operation_costs', {}).get('sentence_bert')
            }
            
            for field, value in metrics_mapping.items():
                if value is not None:
                    setattr(parse_memory, field, value)

        # Now update fields that need validation
        if topics:
            parse_memory.topics = topics
        if emoji_tags:
            parse_memory.emojiTags = emoji_tags
        if emotion_tags:
            parse_memory.emotionTags = emotion_tags
        if memory_chunk_ids:
            parse_memory.memoryChunkIds = memory_chunk_ids
        if steps:
            parse_memory.steps = steps
        
        # Add content if it exists
        if 'content' in updated_memory_item:
            parse_memory.content = updated_memory_item['content']

        # Add other fields from metadata
        metadata_field_mapping = {
            'sourceType': ['sourceType'],
            'context': ['context'],
            'title': ['title'],
            'location': ['location'],
            'hierarchicalStructures': ['hierarchicalStructures', 'hierarchical_structures', 'hierarchical structures'],
            'type': ['type'],
            'sourceUrl': ['sourceUrl'],
            'conversationId': ['conversationId'],
            'steps': ['steps'],
            'current_step': ['current_step'],
            'memoryChunkIds': ['memoryChunkIds']
        }

        # Add fields from metadata if they exist
        for parse_field, meta_fields in metadata_field_mapping.items():
            for meta_field in meta_fields:
                if meta_field in metadata:
                    setattr(parse_memory, parse_field, metadata[meta_field])
                    break

        # Add pointers if they exist
        if "workspace_id" in metadata:
            workspace_id = metadata["workspace_id"]
            if workspace_id and workspace_id.strip():
                parse_memory.workspace = ParsePointer(
                    __type="Pointer",
                    className="WorkSpace",
                    objectId=workspace_id
                )
        
        logger.info(f"parse_memory: {parse_memory}")

        # Convert to dict and remove None values
        data = parse_memory.model_dump(exclude_none=True)
        logger.info(f"Final data after validation: {data}")

        # Clean the data by removing None values and empty collections
        cleaned_data = {
            k: v for k, v in updated_memory_item.items() 
            if v is not None  # Remove None values
            and v != []       # Remove empty lists
            and v != {}       # Remove empty dicts
            and k != 'objectId'  # Don't include objectId in payload
            and k != 'metadata'  # Don't include metadata
        }
        
        logger.info(f"Cleaned update data: {cleaned_data}")

        url = f"{PARSE_SERVER_URL}/parse/classes/Memory/{memory_objectId}?keys=objectId,memoryId,memoryChunkIds,content,updatedAt"
        logger.info(f"URL: {url}")

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            # Update the item
            response = await client.put(
                url,
                headers=headers,
                json=data,
                timeout=30.0
            )
            logger.info(f"update_memory_item response: {response.json()}")

            if response.status_code != 200:
                return UpdateMemoryResponse.error_response(
                    f"Failed to update memory item: {response.text}",
                    code=response.status_code
                )

            # Get the updated item
            get_url = f"{PARSE_SERVER_URL}/parse/classes/Memory/{memory_objectId}"
            get_response = await client.get(get_url, headers=headers)
            
            if get_response.status_code != 200:
                return UpdateMemoryResponse.error_response(
                    f"Failed to get updated memory item: {get_response.text}",
                    code=get_response.status_code
                )

            updated_item = get_response.json()
            logger.info(f"Retrieved updated memory item: {updated_item}")

            # Create UpdateMemoryItem
            memory_item = UpdateMemoryItem(
                objectId=updated_item.get('objectId'),
                memoryId=updated_item.get('memoryId'),
                content=updated_item.get('content'),
                updatedAt=datetime.fromisoformat(updated_item.get('updatedAt').replace('Z', '+00:00')),
                memoryChunkIds=updated_item.get('memoryChunkIds', [])  # Add this line to get memoryChunkIds from Parse
            )

            # Log metrics if present
            if 'metrics' in updated_memory_item:
                logger.info(
                    f"Memory item successfully updated with metrics:\n"
                    f"- Total AI processing cost: ${updated_item.get('totalProcessingCost', 0.0):.6f}\n"
                    f"- Token size: {updated_item.get('tokenSize', 0)}\n"
                    f"- Storage size: {updated_item.get('storageSize', 0)} bytes\n"
                    f"- Operation costs:\n"
                    f"  * Usecase generation: ${updated_item.get('usecaseGenerationCost', 0.0):.6f}\n"
                    f"  * Schema generation: ${updated_item.get('schemaGenerationCost', 0.0):.6f}\n"
                    f"  * Related memories: ${updated_item.get('relatedMemoriesCost', 0.0):.6f}\n"
                    f"  * Node definition: ${updated_item.get('nodeDefinitionCost', 0.0):.6f}\n"
                    f"  * BigBird embedding: ${updated_item.get('bigbirdEmbeddingCost', 0.0):.6f}\n"
                    f"  * Sentence-BERT: ${updated_item.get('sentenceBertCost', 0.0):.6f}"
                )

            # Return successful response with the updated item
            return UpdateMemoryResponse.success_response(
                items=[memory_item],
                status=SystemUpdateStatus(parse=True)  # Only Parse server status is known here
            )

    except Exception as e:
        logger.error(f"Error updating memory item: {e}")
        logger.error("Full traceback:", exc_info=True)
        return UpdateMemoryResponse.error_response(str(e), code=500)

def format_memory_chunk_ids(chunk_ids) -> list:
    """
    Formats memoryChunkIds to ensure proper array format.
    
    Args:
        chunk_ids: The chunk IDs in any format (string, list, etc.)
        
    Returns:
        list: A properly formatted list of chunk IDs
    """
    logger.info(f"memoryChunkIds before formatting: {chunk_ids} (type: {type(chunk_ids)})")
    
    if chunk_ids is None:
        return []
        
    if isinstance(chunk_ids, str):
        try:
            # Try to parse if it's a JSON string
            parsed = json.loads(chunk_ids)
            if isinstance(parsed, list):
                result = [str(id) for id in parsed]
                logger.info(f"memoryChunkIds after formatting: {result} (type: {type(result)})")
                return result
        except json.JSONDecodeError:
            # If it looks like a Python list string but isn't valid JSON
            if chunk_ids.startswith('[') and chunk_ids.endswith(']'):
                # Remove brackets and split on commas
                items = chunk_ids[1:-1].split(',')
                # Clean up each item
                result = [item.strip().strip("'\"") for item in items if item.strip()]
                logger.info(f"memoryChunkIds after formatting: {result} (type: {type(result)})")
                return result
            return []
    
    if isinstance(chunk_ids, list):
        result = [str(id) for id in chunk_ids]
        logger.info(f"memoryChunkIds after formatting: {result} (type: {type(result)})")
        return result
        
    return []

async def upload_file_to_parse(
    file_content: bytes,  # Changed from file_path: str
    filename: str,
    content_type: str,
    session_token: str
) -> Optional[Dict[str, str]]:
    """
    Upload a file to Parse Server and return the file URL and metadata
    
    Args:
        file_content: Content of the file as bytes
        filename: Original filename
        content_type: MIME type of the file
        session_token: Parse session token
    
    Returns:
        Optional[Dict[str, str]]: Dictionary containing file URLs and metadata or None if upload fails
        {
            'file_url': 'https://parsefiles.back4app.com/...',
            'source_url': 'https://parsefiles.back4app.com/...',
            'name': 'original_filename.pdf',
            'mime_type': 'application/pdf'
        }
    """
    try:
        url = f"{PARSE_SERVER_URL}/parse/files/{filename}"
        
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": content_type
        }

        # Upload file to Parse
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                content=file_content,  # Use file_content directly
                timeout=60.0  # Longer timeout for file uploads
            )
            
            if response.status_code == 201:
                response_data = response.json()
                logger.info(f"File uploaded successfully: {response_data}")
                
                return {
                    'file_url': response_data.get('url'),
                    'source_url': response_data.get('url'),
                    'name': response_data.get('name', filename),
                    'mime_type': content_type
                }
            else:
                logger.error(f"Failed to upload file: {response.text}")
                return None

    except Exception as e:
        logger.error(f"Error uploading file to Parse: {e}", exc_info=True)
        return None

async def update_memory_status(
    objectId: str,
    data: dict,
    headers: dict,
    params: dict
) -> dict:
    """Update Memory class status in Parse Server"""
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{PARSE_SERVER_URL}/parse/classes/Memory/{objectId}",
            headers=headers,
            params=params,
            json={k: v for k, v in data.items() if v is not None},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

async def update_post_status(
    post_objectId: str,
    data: dict,
    headers: dict,
    params: dict
) -> dict:
    """Update Post class status in Parse Server"""
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{PARSE_SERVER_URL}/parse/classes/Post/{post_objectId}",
            headers=headers,
            params=params,
            json={k: v for k, v in data.items() if v is not None},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

async def update_document_upload_status(
    objectId: str, 
    filename: str,
    status: DocumentUploadStatusType = DocumentUploadStatusType.PROCESSING,
    progress: float = 0.0,
    current_page: Optional[int] = None,
    total_pages: Optional[int] = None,
    error: Optional[str] = None,
    memory_items: Optional[List[dict]] = None,
    post_objectId: Optional[str] = None,
    upload_id: Optional[str] = None,
    file_url: Optional[str] = None
) -> DocumentUploadStatusResponse:
    """Update document upload status in existing Memory objectId and optionally Post objectId"""
    
    # Prepare update data
    data = {
        "status": status.value,
        "progress": progress,
        "page_number": current_page,
        "total_pages": total_pages,
        "error": error,
        "filename": filename,
        "upload_id": upload_id,
        "file_url": file_url
    }

    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }

    params = {
        "keys": "objectId,status,progress,page_number,total_pages,error,upload_id,user,filename,file_url,file"
    }

    try:
        # Update Memory class
        memory_data = await update_memory_status(objectId, data, headers, params)
        
        # Update Post class if post_objectId is provided
        post_data = None
        if post_objectId:
            post_data = await update_post_status(post_objectId, data, headers, params)
            logger.info(f"Post update response: {post_data}")

        logger.info(f"Successfully updated document upload status for memory {objectId}" + 
                   (f" and post {post_objectId}" if post_objectId else ""))

        # Combine data, preferring memory data but including post file info if available
        response_data = {
            **memory_data,
            **(post_data or {}),
        }

        return DocumentUploadStatusResponse(
            objectId=response_data.get('objectId', objectId),
            status=DocumentUploadStatusType(response_data.get('status', status.value)),
            progress=float(response_data.get('progress', progress)),
            current_page=response_data.get('page_number', current_page),
            total_pages=response_data.get('total_pages', total_pages),
            current_filename=response_data.get('current_filename'),
            error=response_data.get('error', error),
            upload_id=response_data.get('upload_id', ''),
            user=response_data.get('user', {'__type': 'Pointer', 'className': '_User', 'objectId': ''})
        )

    except Exception as e:
        logger.error(f"Failed to update document upload status: {str(e)}")
        return DocumentUploadStatusResponse(
            objectId=objectId,
            filename=filename,
            status=status,
            progress=progress,
            current_page=current_page,
            total_pages=total_pages,
            error=str(e),
            upload_id='',
            user={'__type': 'Pointer', 'className': '_User', 'objectId': ''}
        )

async def get_document_upload_status(
    user_id: str,
    session_token: str,
    upload_id: str
) -> Optional[dict]:
    """Get document upload status from Parse Server"""
    url = f"{PARSE_SERVER_URL}/parse/classes/Memory"
    
    params = {
        "where": json.dumps({
            "type": "DocumentMemoryItem",
            "upload_id": upload_id,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        })
    }

    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if results:
                logger.info(f"Found document upload status for upload_id {upload_id}")
                return results[0]
            logger.info(f"No document upload status found for upload_id {upload_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get document upload status: {str(e)}")
            raise

async def get_post_file_info(post_objectId: str) -> Optional[str]:
    """
    Fetch file information from a Post object in Parse Server.
    
    Args:
        post_objectId (str): The objectId of the Post to fetch
        
    Returns:
        Optional[str]: The file URL if found, None otherwise
        
    Raises:
        ValueError: If no file URL is found in the Post object
    """
    url = f"{PARSE_SERVER_URL}/parse/classes/Post/{post_objectId}"
    
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    params = {
        "keys": "file"  # Only fetch the file field
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            
            post_data = response.json()
            if 'file' in post_data and post_data['file'].get('url'):
                logger.info(f"Successfully retrieved file URL from Post {post_objectId}")
                return post_data['file']['url']
            else:
                raise ValueError(f"No file URL found in Post object {post_objectId}")
                
        except Exception as e:
            logger.error(f"Failed to fetch file info from Post {post_objectId}: {str(e)}")
            raise