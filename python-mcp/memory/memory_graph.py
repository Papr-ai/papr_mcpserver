import sys
from pathlib import Path
import time
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, TYPE_CHECKING, Dict, Any, Optional, Tuple, Set, Union, TypeVar, Awaitable
from pinecone import Pinecone, ServerlessSpec, Vector
from pinecone.core.openapi.data.models import Vector as PineconeVector
import numpy as np
from neo4j.graph import Node as Neo4jNode, Relationship as Neo4jRelationship
from numpy.typing import NDArray
from uuid import uuid4
from models.embedding_model import EmbeddingModel
from memory.memory_item import MemoryItem, memory_item_from_dict
from api_handlers.chat_gpt_completion import ChatGPTCompletion
from services.user import User
from fastapi import APIRouter, BackgroundTasks, Depends
from httpx import AsyncClient
import contextlib 
import asyncio
from services.memory_management import (
    convert_acl, 
    update_memory_item_parse, 
    map_metadata_to_parse_fields,
    add_list_of_usecases,
    add_list_of_goals,
    get_user_goals,
    get_user_usecases,
    get_user_memGraph_schema,
    create_memory_graph,
    extract_goal_titles,
    extract_usecases,
    extract_relationship_types,
    extract_node_names,
    update_memory_item,
    add_list_of_usecases_async,
    add_list_of_goals_async,
    get_user_goals_async,
    get_user_usecases_async,
    get_user_memGraph_schema_async,
    create_memory_graph_async,
    extract_node_names, 
    extract_relationship_types, 
    store_generic_memory_item, 
    convert_neo_item_to_memory_item, 
    flatten_neo_item_to_parse_item, 
    retrieve_memory_item_with_user, 
    delete_memory_item_parse, convert_comma_string_to_list
)
from models.structured_outputs import (
    Node, Relationship, NodeLabel, RelationshipType,
    PersonNode, CompanyNode, CustomerNode, ProjectNode,
    TaskNode, InsightNode, MeetingNode, CodeNode,
    PersonProperties, CompanyProperties, CustomerProperties,
    ProjectProperties, TaskProperties, InsightProperties,
    MeetingProperties, CodeProperties,
    NodeReference, ProcessMemoryResponse, MemoryMetrics, ProcessMemoryData
)
from services.url_utils import clean_url
from models.acl import ACLCondition, ACLFilter
from models.memory_models import MemorySourceInfo, MemoryIDSourceLocation,MemorySourceLocation, NodeConverter, memory_item_to_node, NeoNode, NeoPersonNode, NeoCompanyNode, NeoProjectNode, NeoTaskNode, NeoInsightNode, NeoMeetingNode, NeoOpportunityNode, NeoCodeNode
from typing import Tuple, List, Dict, Any, Optional
from models.neo_path import GraphPath, PathSegment, QueryResult
from models.memory_models import MemoryNodeProperties, RelatedMemoryResult

# Add the 'services' directory to the sys.path
services_dir = str(Path(__file__).parent.parent / 'services')
if services_dir not in sys.path:
    sys.path.append(services_dir)

from models.parse_server import PineconeMatch, ErrorDetail, SystemUpdateStatus, PineconeQueryResponse, ParseStoredMemory, ParseUserPointer, MemoryRetrievalResult, RelatedMemoriesSuccess, RelatedMemoriesError, DeleteMemoryResponse, DeleteMemoryResult, DeletionStatus, UpdateMemoryItem, UpdateMemoryResponse, MemoryParseServerUpdate, ParsePointer
from services.memory_management import store_memory_item, retrieve_memory_item_by_pinecone_id, retrieve_multiple_memory_items,retrieve_memory_item_parse, batch_store_memories, retrieve_memory_items_with_users_async
from memory.memory_item import MemoryItem, TextMemoryItem, CodeSnippetMemoryItem, WebpageMemoryItem, CodeFileMemoryItem, MeetingMemoryItem, PluginMemoryItem, DocumentMemoryItem, IssueMemoryItem, CustomerMemoryItem, memory_item_to_dict
from neo4j import GraphDatabase
from scipy import spatial
from dotenv import find_dotenv, load_dotenv
from os import environ as env
import json
import concurrent.futures
from utils.converter import convert_sets_to_lists
from services.user import User, UserEncoder
from services.logging_config import get_logger
from datastore.neo4jconnection import Neo4jConnection, AsyncNeo4jConnection
import ssl
from typing import Protocol, Any, Dict, Optional
from services.logger_singleton import LoggerSingleton

class AsyncSession(Protocol):
    async def __aenter__(self) -> 'AsyncSession': ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
    async def run(self, query: str, parameters: Optional[Dict[str, Any]] = None): ...
    async def close(self) -> None: ...
    async def begin_transaction(self) -> 'AsyncTransaction': ...

class AsyncTransaction(Protocol):
    async def __aenter__(self) -> 'AsyncTransaction': ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
    async def run(self, query: str, parameters: Optional[Dict[str, Any]] = None): ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

T = TypeVar('T')

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Initialize Neo4j client credentials and pinecone
NEO4J_URL = clean_url(env.get("NEO4J_URL"))
NEO4J_SECRET = clean_url(env.get("NEO4J_SECRET"))

PINECONE_KEY = clean_url(env.get("PINECONE_KEY"))
PINECONE_ENV = clean_url(env.get("PINECONE_ENV"))

#logger.info(f"ENV_FILE: {ENV_FILE}")
logger.debug(f"NEO4J_URL: {NEO4J_URL}")
#logger.info(f"NEO4J_SECRET: {NEO4J_SECRET}")
# Initialize Parse client
parse_application_id = clean_url(env.get("PARSE_APPLICATION_ID"))
parse_rest_api_key = clean_url(env.get("PARSE_REST_API_KEY"))

PARSE_SERVER_URL = clean_url(env.get("PUBLIC_SERVER_URL"))
HEADERS = {
    "X-Parse-Application-Id": parse_application_id,
    "X-Parse-REST-API-Key": parse_rest_api_key,
    "Content-Type": "application/json"
}

api_key = clean_url(env.get("OPENAI_API_KEY"))
organization_id = clean_url(env.get("OPENAI_ORGANIZATION"))


class MemoryGraph:
    def __init__(self, embedding_model=None):
        # Create Neo4j connections (sync and async)
     
        # Initialize async Neo4j connection
        try:
            self.async_neo_conn = AsyncNeo4jConnection(
                uri=NEO4J_URL,
                user="neo4j",
                pwd=NEO4J_SECRET,
                retries=5,
                delay=2
            )
            # Initialize with fallback mode false
            self.async_neo_conn.fallback_mode = False
        except Exception as e:
            logger.error(f"Failed to initialize AsyncNeo4jConnection: {e}")
            # Set to None so ensure_async_connection() can attempt to initialize later
            self.async_neo_conn = None
            # Create connection object in fallback mode
            self.async_neo_conn = AsyncNeo4jConnection(
                uri=NEO4J_URL,
                user="neo4j",
                pwd=NEO4J_SECRET,
                retries=5,
                delay=2
            )
            self.async_neo_conn.fallback_mode = True
            # Initialize fallback storage since Neo4j is unavailable
            self.fallback_storage = {}

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_KEY, environment=PINECONE_ENV)
        self.index = pc.Index("memory-dev")
        self.bigbird_index_name = "memory-bigbird"
        self.bigbird_index = pc.Index(self.bigbird_index_name)
        #self.snowflake_index = pc.Index("memorysnowflake")
        # Replace 'SentenceTransformer' initialization with 'EmbeddingModel'
        # Use provided embedding model or create new one
        self.embedding_model = embedding_model or EmbeddingModel()

        # Initialize memory_items dictionary
        self.memory_items = {}
        
        self.fallback_storage = {}  # Simple in-memory fallback
        
    async def ensure_async_connection(self):
        """Ensure Neo4j connection is established with fallback handling"""
        try:
            if not self.async_neo_conn:
                logger.info("Initializing Neo4j async connection")
                self.async_neo_conn = AsyncNeo4jConnection(
                    uri=NEO4J_URL,
                    user="neo4j",
                    pwd=NEO4J_SECRET
                )
            
            driver = await self.async_neo_conn.get_driver()
            logger.info(f"Retrieved Neo4j driver: {driver}")
            if not driver:
                logger.info("Connection Lost - Initializing Neo4j async connection")
                self.async_neo_conn = AsyncNeo4jConnection(
                    uri=NEO4J_URL,
                    user="neo4j",
                    pwd=NEO4J_SECRET
                )
                driver = await self.async_neo_conn.get_driver()
            if driver:
                try:
                    async with driver.session() as session:
                        result = await session.run("RETURN 1")
                        await result.consume()
                        logger.info("Neo4j connection test successful")
                        # Reset fallback mode if connection is successful
                        if self.async_neo_conn.fallback_mode:
                            logger.info("Resetting fallback mode as connection is restored")
                            self.async_neo_conn.fallback_mode = False
                except ssl.SSLError as ssl_err:
                    logger.error(f"SSL Error connecting to Neo4j: {ssl_err}")
                    logger.error(f"SSL Configuration: CERT_FILE={env.get('SSL_CERT_FILE')}")
                    self.async_neo_conn.fallback_mode = True
                except Exception as e:
                    logger.error(f"Error testing Neo4j connection: {e}")
                    self.async_neo_conn.fallback_mode = True
            else:
                logger.warning("Neo4j connection unavailable, using fallback mode")
                
                self.async_neo_conn.fallback_mode = True
                
                
        except Exception as e:
            logger.error(f"Error ensuring async connection: {str(e)}")
            if self.async_neo_conn:
                self.async_neo_conn.fallback_mode = True
            else:
                # Create connection object in fallback mode if it doesn't exist
                # We still need to create the connection object even in fallback mode
                # so that we have a consistent interface and can track the fallback state
                self.async_neo_conn = AsyncNeo4jConnection(
                    uri=NEO4J_URL,
                    user="neo4j", 
                    pwd=NEO4J_SECRET
                )
                self.async_neo_conn.fallback_mode = True
    
    async def store_with_fallback(self, memory_id: str, data: dict):
        """Store data with fallback mechanism"""
        if self.async_neo_conn.fallback_mode:
            self.fallback_storage[memory_id] = data
            logger.info(f"Stored memory {memory_id} in fallback storage")
            return True
            
        try:
            # Attempt to store in Neo4j
            # return await self._store_in_neo4j(memory_id, data)
            return True
        except Exception as e:
            logger.error(f"Failed to store in Neo4j: {str(e)}")
            self.fallback_storage[memory_id] = data
            return True

    async def recover_fallback_data(self):
        """Attempt to recover data from fallback storage"""
        if not self.fallback_storage:
            return

        logger.info(f"Attempting to recover {len(self.fallback_storage)} items from fallback storage")
        
        try:
            if not self.async_neo_conn.fallback_mode:
                for memory_id, data in self.fallback_storage.items():
                    try:
                        # await self._store_in_neo4j(memory_id, data)
                        # del self.fallback_storage[memory_id]
                        logger.info(f"Recovered memory {memory_id} to Neo4j")
                    except Exception as e:
                        logger.error(f"Failed to recover memory {memory_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Recovery process failed: {str(e)}")

    async def periodic_health_check(self):
        """Periodic health check and recovery attempt"""
        while True:
            try:
                is_healthy = await self.check_neo4j_health()
                if is_healthy and self.fallback_storage:
                   # await self.recover_fallback_data()
                   pass
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                await asyncio.sleep(60)  # Shorter interval if check fails
    
    
    def __iter__(self):
        """Make the MemoryGraph iterable over its memory items"""
        return iter(self.memory_items.values())

    def __len__(self):
        """Return the number of memory items"""
        return len(self.memory_items)

    def get_all_memory_items(self):
        """Get all memory items"""
        return list(self.memory_items.values())
    



    def cosine_similarity(self, vec1, vec2):
        return 1 - spatial.distance.cosine(vec1, vec2)
    
    async def check_and_retrieve_from_pinecone(
        self, 
        session_token: str, 
        embedding: List[float], 
        user_id: str, 
        new_metadata: Dict[str, Any],
        user_workspace_ids: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Asynchronously checks for a similar embedding in Pinecone and updates metadata if a match is found.

        Args:
            session_token (str): The session token for authentication.
            embedding (List[float]): The embedding vector to query.
            user_id (str): The user ID associated with the embedding.
            new_metadata (Dict[str, Any]): The metadata to update in Pinecone if a match is found.

        Returns:
            Optional[str]: The ID of the matched vector if found, else None.
        """
        # Ensure the embedding is a list of floats
        if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
            logger.error(f"Invalid embedding format: {embedding}")
            return None

        # Log the embedding to verify its format
        logger.info(f"Embedding to be queried (first 5 elements): {embedding[:5]}")
        logger.info(f"Embedding length: {len(embedding)}")

        # Ensure the embedding is a list of floats
        embedding = [float(x) for x in embedding]

        # Create a new User instance using the user_id
        user_instance = User.get(user_id)
        
        # Get user roles and workspace IDs
        if user_workspace_ids is None:
            # If workspace IDs weren't provided, fetch both roles and workspaces
            user_roles, user_workspace_ids = await asyncio.gather(
                user_instance.get_roles_async(),
                User.get_workspaces_for_user_async(user_id)
            )
        else:
            # If workspace IDs were provided, only fetch roles
            user_roles = await user_instance.get_roles_async()
        
        logger.debug(f'user_roles {user_roles}')
        logger.debug(f'user_workspace_ids {user_workspace_ids}')

        # Setup the ACL filter using the working structure
        acl_filter = {
            "$or": [
                {"user_id": {"$eq": str(user_id)}},
                {"user_read_access": {"$in": [str(user_id)]}},
                {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}},
                {"role_read_access": {"$in": user_roles}},
            ]
        }

        # Perform a similarity search in Pinecone
        try:
            # Note: Pinecone's query method is synchronous, we'll use asyncio.to_thread if needed
            query_result: PineconeQueryResponse = await asyncio.to_thread(
                self.index.query,
                vector=embedding,
                top_k=1,
                include_values=False,
                include_metadata=True,
                filter=acl_filter
            )


            logger.info(f"Query result: {query_result}")
            if query_result['matches']:
                score = query_result['matches'][0]['score']
                matched_id = query_result['matches'][0]['id']
                logger.info(f"Found match with ID: {matched_id} and similarity score: {score}")
                
                if score > 0.97:
                    logger.info(f"Score {score} > 0.97, using existing vector")
                    # Update the metadata of the matched vector asynchronously
                    logger.debug(f'new_metadata: {new_metadata}')
                    await self.update_memory_metadata(session_token, matched_id, new_metadata)
                    return matched_id
                else:
                    logger.info(f"Score {score} <= 0.97, will create new vector")
                    return None
            else:
                logger.info("No matches found in Pinecone")
                return None

        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
        return None

    async def update_pinecone_metadata(self, vector_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Asynchronously updates the metadata of an existing vector in Pinecone.

        Args:
            vector_id (str): The ID of the vector to update.
            new_metadata (Dict[str, Any]): The new metadata to set.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            # Fetch existing metadata - using asyncio.to_thread since Pinecone SDK is sync
            fetch_result = await asyncio.to_thread(self.index.fetch, ids=[vector_id])
            vector_data = fetch_result.vectors.get(vector_id, {})
            existing_metadata = vector_data.metadata or {}
            existing_vector = vector_data.values or []

            # Merge the existing metadata with the new metadata
            updated_metadata = {**existing_metadata, **new_metadata}

            # Update the metadata in Pinecone
            await asyncio.to_thread(
                self.index.upsert,
                vectors=[(vector_id, existing_vector, updated_metadata)]
            )

            logger.info(f"Successfully updated metadata for vector ID: {vector_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating metadata in Pinecone for vector ID {vector_id}: {e}")
            return False
    
    async def update_neo_metadata(self, memory_id: str, new_metadata: Dict[str, Any]) -> bool:
        try:
            await self.ensure_async_connection()
            
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, storing metadata in fallback storage")
                #self.fallback_storage[memory_id] = new_metadata
                return True

            # Flatten the metadata dictionary if necessary
            flat_metadata = self.validate_metadata(new_metadata)

            # Get base ID (remove chunk suffix if present)
            base_memory_id = memory_id.split('_')[0]

            # Main update query - modified to return all properties explicitly
            update_query = """
                MATCH (m:Memory)
                WHERE m.id = $memory_id 
                OR m.id = $base_memory_id
                OR $memory_id IN coalesce(m.memoryChunkIds, [])
                OR $base_memory_id IN coalesce(m.memoryChunkIds, [])
                SET m += $flat_metadata
                RETURN properties(m) as props, m.id as id, labels(m) as labels
            """

            async with self.async_neo_conn.get_session() as session:
                # Run update query with all necessary parameters
                result = await session.run(
                    update_query,
                    memory_id=str(memory_id),
                    base_memory_id=str(base_memory_id),
                    flat_metadata=flat_metadata
                )
                updated_record = await result.single()
                await result.consume()

                if updated_record and updated_record.get("props"):
                    props = updated_record.get("props", {})
                    node_id = props.get("id") or updated_record.get("id")  # Try both locations
                    labels = updated_record.get("labels", [])
                    logger.info(f"Successfully updated node with ID: {node_id}")
                    logger.info(f"Node properties: {props}")
                    logger.info(f"Node labels: {labels}")
                    return True
                
                # If we get here, try to verify if the node exists
                verify_query = """
                MATCH (m:Memory)
                WHERE m.id = $base_memory_id
                RETURN properties(m) as props
                """
                verify_result = await session.run(verify_query, base_memory_id=str(base_memory_id))
                verify_record = await verify_result.single()
                await verify_result.consume()

                if verify_record and verify_record.get("props"):
                    logger.info(f"Node exists but update didn't return data. Node properties: {verify_record.get('props')}")
                    return True

                logger.error(f"No memory item found with ID: {memory_id} or base ID: {base_memory_id} in Neo4j")
                logger.error(f"Debug: memory_id={memory_id}, base_memory_id={base_memory_id}")
                return False

        except Exception as e:
            logger.error(f"Error updating metadata in Neo4j for memory ID {memory_id}: {e}")
            # Store in fallback storage if Neo4j update fails
            self.fallback_storage[memory_id] = new_metadata
            return True


    async def update_parse_metadata(
        self, 
        session_token: str, 
        memory_id: str, 
        new_metadata: Dict[str, Any]
    ) -> bool:
        """
        Asynchronously updates the metadata of an existing memory item in Parse Server.

        Args:
            session_token (str): The session token for authentication.
            memory_id (str): The unique ID of the memory item to update.
            new_metadata (Dict[str, Any]): The new metadata to set.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            # Extract memoryChunkIds from new_metadata if it exists
            memory_chunk_ids = new_metadata.get('memoryChunkIds', [])
            if isinstance(memory_chunk_ids, str):
                try:
                    memory_chunk_ids = json.loads(memory_chunk_ids)
                except json.JSONDecodeError:
                    memory_chunk_ids = [id.strip() for id in memory_chunk_ids.strip('[]').split(',') if id.strip()]
            
            # Ensure memory_chunk_ids is a list of strings
            if isinstance(memory_chunk_ids, list):
                memory_chunk_ids = [str(id).strip() for id in memory_chunk_ids if id]
            else:
                memory_chunk_ids = []
                
            # Retrieve the existing memory item
            memory_item: ParseStoredMemory = await retrieve_memory_item_parse(
                session_token=session_token, 
                memory_item_id=memory_id,
                memory_chunk_ids=memory_chunk_ids
            )
            
            if not memory_item:
                logger.error(f"No memory item found with ID: {memory_id.split('_')[0]} in Parse Server")
                return False

            # Convert custom ACL to Parse ACL
            parse_acl = convert_acl(new_metadata)

            # Pre-process list fields
            emoji_tags = convert_comma_string_to_list(
                new_metadata.get('emojiTags') or 
                new_metadata.get('emoji_tags') or 
                new_metadata.get('emoji tags')
            )
            emotion_tags = convert_comma_string_to_list(
                new_metadata.get('emotionTags') or 
                new_metadata.get('emotion_tags') or 
                new_metadata.get('emotion tags')
            )
            topics = convert_comma_string_to_list(new_metadata.get('topics'))
            steps = convert_comma_string_to_list(new_metadata.get('steps'))

            # Create update data using MemoryParseServerUpdate
            parse_memory = MemoryParseServerUpdate(
                ACL=parse_acl,
                sourceType=new_metadata.get('sourceType'),
                context=new_metadata.get('context'),
                title=new_metadata.get('title'),
                location=new_metadata.get('location'),
                emojiTags=emoji_tags,
                emotionTags=emotion_tags,
                hierarchicalStructures=(
                    new_metadata.get('hierarchicalStructures') or 
                    new_metadata.get('hierarchical_structures') or 
                    new_metadata.get('hierarchical structures')
                ),
                sourceUrl=new_metadata.get('sourceUrl'),
                conversationId=new_metadata.get('conversationId'),
                topics=topics,
                steps=steps,
                current_step=new_metadata.get('current_step'),
                memoryChunkIds=memory_chunk_ids
            )


            # Convert to dict and remove None values
            update_data = parse_memory.model_dump(
                exclude_none=True,
                exclude={'createdAt', 'updatedAt'}
            )

            logger.info(f"Data to update in Parse Server: {update_data} with objectId: {memory_item.objectId}")

            # Perform the update
            success = await update_memory_item_parse(
                session_token=session_token,
                object_id=memory_item.objectId,
                update_data=update_data
            )

            if success:
                logger.info(f"Successfully updated memory item with ID: {memory_id.split('_')[0]}")
                return True
            else:
                logger.error(f"Failed to update memory item with ID: {memory_id.split('_')[0]}")
                return False

        except Exception as e:
            logger.error(f"Error updating memory item with ID {memory_id.split('_')[0]}: {e}")
            return False

    async def update_memory_metadata(
        self,
        session_token: str, 
        vector_id: str, 
        new_metadata: Dict[str, Any]
    ) -> bool:
        """
        Asynchronously updates metadata across Pinecone, Neo4j, and Parse Server for a given memory item.

        Args:
            session_token (str): The session token for authentication.
            vector_id (str): The ID of the vector in Pinecone to update.
            new_metadata (Dict[str, Any]): The new metadata to set.

        Returns:
            bool: True if all updates were successful, False otherwise.
        """
        try:
            # Update metadata in all systems concurrently
            await asyncio.gather(
                self.update_pinecone_metadata(vector_id, new_metadata),
                self.update_neo_metadata(vector_id, new_metadata),
                self.update_parse_metadata(session_token, vector_id, new_metadata)
            )

            logger.info(f"Successfully updated metadata across Pinecone, Neo4j, and Parse Server for vector ID: {vector_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating metadata across systems for vector ID {vector_id}: {e}")
            return False
        
    async def add_memory_item_without_relationships(
        self, 
        session_token: str, 
        memory_item: MemoryItem, 
        relationships_json: Optional[dict] = None,
        user_workspace_ids: Optional[List[str]] = None
    ) -> Tuple[List[ParseStoredMemory], List[MemoryItem]]:
        """
        Asynchronously adds a memory item to storage without processing relationships.
        
        Args:
            session_token (str): The session token for authentication
            memory_item (MemoryItem): The memory item to store
            relationships_json (Optional[dict]): Optional relationships data
            
        Returns:
            Tuple[List[ParseStoredMemory], List[MemoryItem]]: A tuple containing:
                - List of stored memory properties from Parse Server
                - List of processed memory items
                - List of user workspace ids
        """
        # Initialize with proper types
        added_item_properties_list: List[ParseStoredMemory] = []
        memory_item_list: List[MemoryItem] = []        
        
        total_start_time = time.time()
        timings = {
            'metadata_prep': 0,
            'embedding_generation': 0,
            'similarity_check': 0,
            'pinecone_store': 0,
            'parse_server_store': 0,
            'neo4j_store': 0,
            'total': 0
        }

        # Metadata preparation timing
        metadata_start = time.time()
        if 'createdAt' not in memory_item.metadata:
            memory_item.metadata['createdAt'] = datetime.utcnow().isoformat()
        
        def sanitize_metadata(metadata):
            return {
                key: 'None' if value is None else value 
                for key, value in metadata.items() 
                if key != 'text'
            }
        
        memory_item.metadata = sanitize_metadata(memory_item.metadata)
        metadata_to_upsert = memory_item.metadata.copy()
        logger.info(f'memory_item metadata {metadata_to_upsert} with type {type(metadata_to_upsert)}')
        timings['metadata_prep'] = time.time() - metadata_start
        logger.info(f"Metadata preparation took {timings['metadata_prep']:.4f} seconds")

        # Extract user and workspace information
        user_id = memory_item.metadata.get('user_id')
        workspace_id = memory_item.metadata.get('workspace_id')
        logger.info(f'user_id: {user_id}')
        logger.info(f'workspace_id: {workspace_id}')

        # Generate embeddings for all chunks with retry mechanism
        embedding_start = time.time()
        max_retries = 5
        retry_delay = 1  # Initial delay in seconds
        
        async def try_generate_embeddings():
            try:
                logger.info(f'Generating embeddings for memory item: {memory_item.content}')
                embeddings, chunks = await self.embedding_model.get_sentence_embedding(memory_item.content)
                logger.info(f'Generated {len(embeddings)} embeddings from {len(chunks)} chunks')
                return embeddings, chunks
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                return None, None

        for attempt in range(max_retries):
            embeddings, chunks = await try_generate_embeddings()
            
            if embeddings is not None and chunks is not None:
                break
                
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                retry_delay_with_backoff = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {retry_delay_with_backoff}s")
                await asyncio.sleep(retry_delay_with_backoff)
        
        if embeddings is None or chunks is None:
            logger.error("Failed to generate embeddings after all retries")
            return {
                "status_code": 500,
                "success": False,
                "error": "Failed to generate embeddings after multiple attempts",
                "data": None
            }

        # Ensure embeddings are lists of floats
        embeddings = [list(map(float, embedding)) for embedding in embeddings]
        timings['embedding_generation'] = time.time() - embedding_start
        logger.info(f"Embedding generation took {timings['embedding_generation']:.4f} seconds")

        memoryChunkIds = []  # Track all chunk IDs
        logger.info(f"Initializing empty memoryChunkIds list")
        existing_main_id = None  # Track the first existing ID we find

        # Check existing chunks concurrently
        similarity_start = time.time()
        existing_memory_ids = await asyncio.gather(*[
            self.check_and_retrieve_from_pinecone(session_token, emb, user_id, metadata_to_upsert, user_workspace_ids)
            for emb in embeddings
        ])
        timings['similarity_check'] = time.time() - similarity_start
        logger.info(f"Similarity checks took {timings['similarity_check']:.4f} seconds")
        logger.info(f'existing_memory_ids: {existing_memory_ids}')
        
        # Process existing chunks and prepare new chunks
        new_chunks = []
        for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            existing_id = existing_memory_ids[idx]
            chunk_metadata = metadata_to_upsert.copy()
            chunk_metadata.update({
                'chunk_index': idx,
                'total_chunks': len(chunks)
            })

            if existing_id:
                logger.info(f'Memory chunk with similar embedding exists in Pinecone with id {existing_id}')
                # Get the base ID (remove chunk suffix if present)
                base_id = existing_id.split('_')[0]
                logger.info(f'Using base ID: {base_id} from chunk ID: {existing_id}')
                
                # First check Neo4j for the base ID
                neo4j_item = await self.get_memory_item(base_id)
                if neo4j_item:
                    logger.info(f'Found existing memory in Neo4j with ID: {base_id}')
                    
                    # Since it exists in Neo4j, check Parse Server
                    existing_parse_item = await retrieve_memory_item_by_pinecone_id(
                        session_token, 
                        base_id  # Use base_id, not chunk ID
                    )
                    
                    if existing_parse_item:
                        logger.info(f'Found existing memory in Parse Server with ID: {base_id}')
                        logger.info(f'existing_parse_item: {existing_parse_item}')
                        logger.info(f'existing_parse_item memoryChunkIds: {existing_parse_item.get("memoryChunkIds")}')
                        logger.info(f'existing_parse_item metadata: {existing_parse_item.get("metadata")}')
                        # Extract memoryChunkIds from existing item
                        existing_chunk_ids = []
                        if isinstance(existing_parse_item.get('metadata'), str):
                            try:
                                metadata = json.loads(existing_parse_item['metadata'])
                                existing_chunk_ids = metadata.get('memoryChunkIds', [])
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse metadata JSON for existing memory {base_id}")
                        elif isinstance(existing_parse_item.get('metadata'), dict):
                            existing_chunk_ids = existing_parse_item['metadata'].get('memoryChunkIds', [])
                        
                        # Also check direct memoryChunkIds field
                        if not existing_chunk_ids and existing_parse_item.get('memoryChunkIds'):
                            existing_chunk_ids = existing_parse_item['memoryChunkIds']
                        
                        # For legacy memories, use the memoryId as the chunk ID if no chunks found
                        if not existing_chunk_ids:
                            existing_chunk_ids = [base_id]  # Use base_id as the single chunk ID
                            logger.info(f"Legacy memory found - using memoryId as chunk ID: {existing_chunk_ids}")
                            
                        logger.info(f"Found existing chunk IDs: {existing_chunk_ids}")
                        # Create a properly formatted ParseStoredMemory object
                        parse_stored_memory = ParseStoredMemory(
                            objectId=existing_parse_item.get('objectId'),
                            createdAt=existing_parse_item.get('createdAt'),
                            updatedAt=existing_parse_item.get('updatedAt'),
                            ACL=existing_parse_item.get('ACL', {}),
                            content=existing_parse_item.get('content', ''),
                            type=existing_parse_item.get('type', 'text'),
                            metadata=existing_parse_item.get('metadata', '{}'),
                            memoryId=base_id,  # Use base_id
                            memoryChunkIds=existing_chunk_ids,  
                            user=ParseUserPointer(
                                objectId=existing_parse_item.get('user', {}).get('objectId'),
                                className='_User'
                            )
                        )
                        
                        # Update memory item with Parse Server data
                        memory_item.objectId = existing_parse_item.get('objectId')
                        memory_item.createdAt = existing_parse_item.get('createdAt')
                        memory_item.id = base_id  # Ensure consistent ID
                        
                        added_item_properties_list.append(parse_stored_memory)
                        memory_item_list.append(memory_item)
                        logger.info(f'Returning existing memory item from all systems with ID: {base_id}')
                        return added_item_properties_list, memory_item_list
                
                # If we didn't find it in Neo4j/Parse, continue with new item creation
                logger.info(f'Memory {base_id} not found in Neo4j/Parse, will create new')
                memory_item.id = base_id  # Use the base_id for consistency

            # Only add new chunk if we didn't find existing item in Neo4j/Parse
            chunk_id = str(memory_item.id) + f"_{idx}"  # Ensure clean string format from the start
            new_chunks.append((chunk_id, embedding, chunk_metadata))
            memoryChunkIds.append(chunk_id)  # Add clean string ID
            logger.info(f"Added chunk_id to memoryChunkIds: {chunk_id}")
            logger.info(f"Current memoryChunkIds list: {memoryChunkIds}")

        # Batch upsert new chunks to Pinecone
        if new_chunks:
            pinecone_start = time.time()
            try:
                # Prepare vectors in the format expected by upsert_batch
                vectors_batch = [{
                    'id': chunk_id,
                    'values': embedding,
                    'metadata': metadata
                } for chunk_id, embedding, metadata in new_chunks]
                
                logger.info(f"Preparing to upsert vectors to Pinecone with IDs: {[v['id'] for v in vectors_batch]}")
                
                # Use upsert_batch for more efficient batch operations
                await asyncio.to_thread(
                    self.index._upsert_batch, 
                    vectors=vectors_batch,
                    namespace=None,
                    _check_type=True
                )
                
                # Verify the vectors were added by querying Pinecone with retries
                max_retries = 15
                retry_delay = 1  # seconds between retries

                for vector in vectors_batch:
                    verified = False
                    attempts = 0
                    
                    while not verified and attempts < max_retries:
                        try:
                            attempts += 1
                            verify_result = await asyncio.to_thread(
                                self.index.fetch,
                                ids=[vector['id']]
                            )
                            
                            if vector['id'] in verify_result['vectors']:
                                logger.info(f"Successfully verified vector {vector['id']} in Pinecone (attempt {attempts})")
                                verified = True
                                break
                            else:
                                logger.warning(f"Vector {vector['id']} not found in Pinecone (attempt {attempts})")
                                if attempts < max_retries:
                                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2  # Exponential backoff
                        except Exception as e:
                            logger.error(f"Error verifying vector {vector['id']} (attempt {attempts}): {e}")
                            if attempts < max_retries:
                                logger.info(f"Waiting {retry_delay} seconds before retry...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff

                    if not verified:
                        logger.error(f"Failed to verify vector {vector['id']} after {max_retries} attempts")
                        raise Exception(f"Vector {vector['id']} not found in Pinecone after {max_retries} verification attempts")

                logger.info(f'Successfully upserted and verified {len(new_chunks)} new chunks to Pinecone')
                
            except Exception as e:
                logger.error(f"Failed to upsert or verify chunks in Pinecone: {e}", exc_info=True)
                return None, None
                
            timings['pinecone_store'] = time.time() - pinecone_start
            logger.info(f"Pinecone storage and verification took {timings['pinecone_store']:.4f} seconds")

        # Store single node in Neo4j with all chunk IDs
        neo4j_start = time.time()
        try:
            memory_item.metadata['memoryChunkIds'] = memoryChunkIds
            memory_item.memoryChunkIds = memoryChunkIds  # Add this line to set it directly on the object
            logger.info(f"Storing memory item with chunk IDs: {memoryChunkIds}")
            logger.info(f"Preparing to store in Neo4j with memoryChunkIds: {memoryChunkIds}")
            await self.add_memory_item_to_neo4j(memory_item, memoryChunkIds)
            timings['neo4j_store'] = time.time() - neo4j_start
            logger.info(f"Neo4j storage took {timings['neo4j_store']:.4f} seconds")
        except Exception as e:
            logger.error(f"Failed to add memory node to Neo4j: {e}", exc_info=True)

        # Store in Parse Server with all chunk IDs
        parse_start = time.time()
        try:
            # Ensure memoryChunkIds are in metadata before storing
            memory_item.metadata['memoryChunkIds'] = memoryChunkIds
            memory_item.memoryChunkIds = memoryChunkIds  # Add this line to set it directly on the object
            logger.info(f"Storing memory item with chunk IDs: {memoryChunkIds}")
            logger.info(f"Preparing to store in Parse Server with memoryChunkIds: {memoryChunkIds}")
            logger.info(f"Memory item metadata before Parse Server storage: {memory_item.metadata}")
            
            added_item_properties = await store_memory_item(user_id, session_token, memory_item)
            logger.info(f"Calling store_memory_item with memory_item metadata: {memory_item.metadata}")
            
            if not added_item_properties:
                logger.error("Failed to store memory item in Parse server")
            else:
                # Add detailed logging of the response before conversion
                logger.info(f"Raw added_item_properties before conversion: {added_item_properties}")
                logger.info(f"Raw added_item_properties type: {type(added_item_properties)}")
                if isinstance(added_item_properties, dict):
                    logger.info(f"Dict keys: {added_item_properties.keys()}")
                    logger.info(f"Metadata in response: {added_item_properties.get('metadata', 'No metadata')}")
                
                # Check if added_item_properties is a Pydantic model
                if hasattr(added_item_properties, 'model_copy'):
                    logger.info("Processing Pydantic model response")
                    updated_properties = added_item_properties.model_copy(
                        update={
                            'memoryId': str(memory_item.id),
                            'memoryChunkIds': memoryChunkIds  # Ensure chunk IDs are included
                        }
                    )
                    logger.info(f"Updated Pydantic properties: {updated_properties}")
                    added_item_properties_list.append(updated_properties)
                else:
                    # Handle dictionary response
                    logger.info("Processing dictionary response")
                    updated_properties = {
                        **added_item_properties,
                        'memoryId': str(memory_item.id),
                        'memoryChunkIds': memoryChunkIds  # Ensure chunk IDs are included
                    }
                    logger.info(f"Final properties before ParseStoredMemory creation: {updated_properties}")
                    added_item_properties_list.append(ParseStoredMemory(**updated_properties))
                    
                memory_item_list.append(memory_item)
                logger.info(f'Added memory item with id {memory_item.id} and chunk IDs {memoryChunkIds} to Parse Server')
        except Exception as e:
            logger.error(f"Failed to store in Parse Server: {e}", exc_info=True)
            # Return empty lists instead of None to maintain return type consistency
            return [], []
            
        timings['parse_server_store'] = time.time() - parse_start
        logger.info(f"Parse Server storage took {timings['parse_server_store']:.4f} seconds")

        # Calculate total time
        timings['total'] = time.time() - total_start_time
        logger.info("Memory item processing timings:")
        for operation, duration in timings.items():
            logger.info(f"  {operation}: {duration:.4f} seconds")

        return added_item_properties_list, memory_item_list

    async def update_memory_item_with_relationships(
        self, 
        memory_item: 'MemoryItem',  # Use string literal for forward reference
        relationships_json: List[Dict[str, str]], 
        workspace_id: Optional[str], 
        user_id: str
    ) -> Dict[str, Union[bool, List[Relationship], Optional[str]]]:
        """
        Creates relationships between memory items and returns relationship status.
        
        Returns:
            Dict containing:
            - success (bool): Whether all relationships were created successfully
            - relationships (List[Relationship]): List of created relationships
            - error (Optional[str]): Error message if any
        """
        memory_item_ids: List[str] = []
        created_relationships: List[Relationship] = []
        success = True
        error = None

        try:
            # Check if 'context' key exists in memory_item
            # Only check context if it exists and is not empty in the memory_dict
            if memory_item.context and len(memory_item.context) > 0:
                memory_item_ids = await self.get_memory_item_ids_from_conversation_history(memory_item.context, user_id)
                logger.info(f'Got memory item ids from conversation history: {memory_item_ids}')
            else:
                logger.info("No context provided in memory_dict, skipping conversation history processing.")

            # Iterate over the relationships in the JSON structure
            for relationship in relationships_json:
                try:
                    # Extract relationship data
                    related_item_id = str(relationship.get('related_item_id', ''))
                    relation_type = str(relationship.get('relation_type', ''))

                    if not related_item_id or not relation_type:
                        logger.warning(f"Skipping invalid relationship: {relationship}")
                        continue

                    # Handle special relationship cases
                    if related_item_id == 'previous_memory_item_id':
                        if memory_item_ids:
                            related_item_id = str(memory_item_ids[-1])
                            logger.info(f'Using most recent memory item ID: {related_item_id}')
                        else:
                            logger.info('No previous memory items found, skipping relationship')
                            continue

                    elif related_item_id == 'all_previous_memory_items':
                        for prev_memory_id in memory_item_ids:
                            success &= await self._create_single_relationship(
                                str(prev_memory_id),
                                str(memory_item.id),
                                relation_type,
                                workspace_id,
                                user_id,
                                created_relationships
                            )
                        continue

                    # Create regular relationship
                    relationship_success = await self._create_single_relationship(
                        str(memory_item.id),
                        related_item_id,
                        relation_type,
                        workspace_id,
                        user_id,
                        created_relationships
                    )
                    success = success and relationship_success

                except Exception as rel_error:
                    logger.error(f"Error processing relationship {relationship}: {rel_error}")
                    success = False
                    continue

        except Exception as e:
            logger.error(f"Error creating relationships: {e}")
            success = False
            error = str(e)

        return {
            "success": success and len(created_relationships) > 0,
            "relationships": created_relationships,
            "error": error
        }

    async def _create_single_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        workspace_id: Optional[str],
        user_id: str,
        created_relationships: List[Relationship]
    ) -> bool:
        """Helper method to create a single relationship and update the relationships list"""
        try:
            result = await self.link_memory_items_async(
                source_id,
                target_id,
                relation_type,
                workspace_id,
                user_id
            )
            
            if result:
                created_relationships.append(Relationship(
                    type=relation_type,
                    direction="->",
                    source=NodeReference(
                        label="Memory",
                        id=source_id
                    ),
                    target=NodeReference(
                        label="Memory",
                        id=target_id
                    )
                ))
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error creating relationship between {source_id} and {target_id}: {e}")
            return False

    async def memory_item_exists_async(self, session, memory_item_id: str) -> bool:
        """Async check if memory item exists in Neo4j"""
        try:
            result = await session.run(
                "MATCH (a:Memory) WHERE a.id = $id RETURN a",
                id=memory_item_id
            )
            record = await result.single()
            await result.consume()
            return record is not None
        except Exception as e:
            logger.error(f"Error checking memory item existence: {e}")
            return False
    
    def _strip_chunk_id(self, memory_id: str) -> str:
        """Strips chunk identifier (_0, _1, etc.) from memory ID."""
        if not memory_id:
            return memory_id
        return memory_id.split('_')[0]

    async def link_memory_items_async(
        self, 
        item_id_1: str, 
        item_id_2: str, 
        relation_type: str, 
        workspace_id: Optional[str] = None, 
        user_id: Optional[str] = None
    ) -> bool:
        """Async version of link_memory_items"""
        try:
            # Strip chunk identifiers from memory IDs
            clean_id_1 = self._strip_chunk_id(str(item_id_1))
            clean_id_2 = self._strip_chunk_id(str(item_id_2))
            
            logger.info(f"Original IDs: {item_id_1}, {item_id_2}")
            logger.info(f"Cleaned IDs: {clean_id_1}, {clean_id_2}")

            # Ensure Neo4j connection is initialized
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, cannot create relationships")
                return False

            async with self.async_neo_conn.get_session() as session:
                # First verify both nodes exist
                verify_query = """
                MATCH (a:Memory)
                WHERE a.id = $item_id_1
                WITH a
                MATCH (b:Memory)
                WHERE b.id = $item_id_2
                RETURN a.id as id1, b.id as id2
                """
                parameters = {'item_id_1': str(clean_id_1), 'item_id_2': str(clean_id_2)}

                logger.info(f"Executing verify query with parameters: {parameters}")
                result = await session.run(verify_query, parameters)
                record = await result.single()
                logger.info(f"Verify query record: {record}")
                await result.consume()

                # Check if both nodes were found
                if not record or not record.get("id1") or not record.get("id2"):
                    error_msg = f"Memory items {clean_id_1} and/or {clean_id_2} not found, skipping relationship"
                    logger.error(error_msg)
                    return False

                # If we get here, both nodes exist, create relationship
                create_query = f"""
                MATCH (a:Memory)
                WHERE a.id = $item_id_1
                MATCH (b:Memory)
                WHERE b.id = $item_id_2
                MERGE (a)-[r:{relation_type} {{
                    workspace_id: $workspace_id,
                    user_id: $user_id,
                    type: $relation_type,
                    created_at: datetime()
                }}]->(b)
                RETURN type(r) as rel_type
                """
                parameters = {
                    'item_id_1': str(clean_id_1),
                    'item_id_2': str(clean_id_2),
                    'relation_type': relation_type,
                    'workspace_id': workspace_id,
                    'user_id': user_id
                }
                logger.info(f"Executing create query with parameters: {parameters}")
                result = await session.run(create_query, parameters)
                record = await result.single()
                logger.info(f"Create query record: {record}")
                await result.consume()

                # Check if relationship was created
                if not record or not record.get("rel_type"):
                    error_msg = f"Failed to create relationship between {clean_id_1} and {clean_id_2}"
                    logger.error(error_msg)
                    return False
                
                logger.info(f"Successfully created {relation_type} relationship between {clean_id_1} and {clean_id_2}")
                return True

        except Exception as e:
            logger.error(f"Error in link_memory_items_async: {str(e)}")
            return False

    
    async def add_memory_item_to_bigbird(self, memory_item: Dict[str, Any], related_memories: List[ParseStoredMemory]):
        """
        Add memory item and its related memories to BigBird index asynchronously.
        
        Args:
            memory_item (Dict[str, Any]): The primary memory item to index
            related_memories (List[ParseStoredMemory]): List of related memory items
        """
        # Start with the original memory_item's content
        aggregated_text = memory_item['content']
        
        # Append content from related memories
        for related_memory in related_memories:
            aggregated_text += " " + related_memory.content
        
        # Get the embedding for aggregated text
        embedding, chunks = await self.embedding_model.get_bigbird_embedding(aggregated_text)
        memory_item_lst = []

        # Get related memory IDs
        related_memory_ids = [memory.memoryId for memory in related_memories]
        
        # Prepare metadata
        memory_item_metadata = memory_item.get('metadata', {})
        if isinstance(memory_item_metadata, dict):
            memory_item_metadata['relatedMemoryIds'] = related_memory_ids
        else:
            memory_item_metadata = {'relatedMemoryIds': related_memory_ids}

        # Prepare batch vectors with proper typing
        vectors_to_upsert = []
        
        # Add main memory item
        memory_item_id = str(memory_item['id'])  # Ensure ID is string
        
        # Convert embedding to list if it's numpy array
        first_embedding: List[float] = embedding[0]
        if isinstance(first_embedding, np.ndarray):
            first_embedding = first_embedding.tolist()
        elif isinstance(first_embedding, list):
            first_embedding = [float(x) for x in first_embedding]
        else:
            raise ValueError(f"Unexpected embedding type: {type(first_embedding)}")

        # Create vector with string ID
        vector = PineconeVector(
            id=str(memory_item_id),  # Ensure ID is string
            values=first_embedding,
            metadata=memory_item_metadata
        )
        
        vectors_to_upsert.append(vector)
        memory_item_lst.append(str(memory_item_id))  # Ensure ID is string

        # Process chunks
        chunk_type = memory_item['type']
        chunk_metadata = memory_item_metadata
        chunk_context = memory_item.get('context', [])
        chunk_relationships_json = memory_item.get('relationships_json', [])

        # Prepare all chunk vectors
        for index in range(1, len(embedding)):
            try:
                memory_item_chunk = None
                if chunk_type == 'TextMemoryItem':
                    memory_item_chunk = TextMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'CodeSnippetMemoryItem':
                    memory_item_chunk = CodeSnippetMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'DocumentMemoryItem':
                    memory_item_chunk = DocumentMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'WebpageMemoryItem':
                    memory_item_chunk = WebpageMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'CodeFileMemoryItem':
                    memory_item_chunk = CodeFileMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'MeetingMemoryItem':
                    memory_item_chunk = MeetingMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'PluginMemoryItem':
                    memory_item_chunk = PluginMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'IssueMemoryItem':
                    memory_item_chunk = IssueMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                elif chunk_type == 'CustomerMemoryItem':
                    memory_item_chunk = CustomerMemoryItem(chunks[index], chunk_metadata, chunk_context, relationships_json=chunk_relationships_json)
                else:
                    logger.error(f"Unknown memory item type: {chunk_type}")
                    continue

                if memory_item_chunk:
                    vectors_to_upsert.append((
                        memory_item_chunk.id,
                        embedding[index],
                        chunk_metadata
                    ))
                    memory_item_lst.append(memory_item_chunk.id)

            except Exception as e:
                logger.error(f"Error creating memory chunk of type {chunk_type}: {str(e)}")
                continue

        # Use asyncio.to_thread for the batch upsert
        try:
            # Split into batches of 100 (Pinecone's recommended batch size)
            batch_size = 100
            max_retries = 3
            base_delay = 1  # Base delay in seconds

            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                retry_count = 0
                last_exception = None

                while retry_count < max_retries:
                    try:
                        # Convert batch to correct format
                        processed_batch = []
                        for vec in batch:
                            if isinstance(vec, PineconeVector):
                                # Handle PineconeVector format
                                processed_batch.append({
                                    'id': str(vec.id),
                                    'values': [float(v) for v in vec.values],
                                    'metadata': vec.metadata
                                })
                            elif isinstance(vec, tuple):
                                # Handle tuple format (id, embedding, metadata)
                                processed_batch.append({
                                    'id': str(vec[0]),
                                    'values': [float(v) for v in vec[1]],
                                    'metadata': vec[2]
                                })
                            else:
                                logger.warning(f"Skipping vector with unexpected format: {type(vec)}")
                                continue

                        if processed_batch:
                            await asyncio.to_thread(
                                self.bigbird_index.upsert,
                                vectors=processed_batch,
                                namespace=None
                            )
                            logger.info(f'Successfully upserted batch of {len(processed_batch)} vectors to BigBird Pinecone index')
                            break  # Success - exit retry loop
                        
                    except Exception as e:
                        last_exception = e
                        retry_count += 1
                        if retry_count < max_retries:
                            # Calculate delay with exponential backoff
                            delay = base_delay * (2 ** (retry_count - 1))  # 1s, 2s, 4s
                            logger.warning(
                                f"Batch upsert failed (attempt {retry_count}/{max_retries}). "
                                f"Retrying in {delay}s. Error: {str(e)}"
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                f"Failed to upsert batch after {max_retries} attempts. "
                                f"Final error: {str(last_exception)}",
                                exc_info=True
                            )
                            # Continue with next batch instead of failing completely
                            break

        except Exception as e:
            logger.error(f"Error in batch upserting to Pinecone: {str(e)}")
            raise

        return memory_item_lst

    async def update_pinecone(
        self, 
        vector_id: str, 
        embedding: List[float], 
        new_metadata: Dict[str, Any]
    ) -> bool:
        """
        Asynchronously updates both the embedding and metadata of an existing vector in Pinecone.

        Args:
            vector_id (str): The ID of the vector to update
            embedding (List[float]): The new embedding vector
            new_metadata (Dict[str, Any]): The new metadata to set

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Fetch existing metadata to merge - using asyncio.to_thread since Pinecone SDK is sync
            fetch_result = await asyncio.to_thread(self.index.fetch, ids=[vector_id])
            if not fetch_result or vector_id not in fetch_result.get('vectors', {}):
                logger.error(f"No vector found with ID {vector_id}")
                return False

            vector_data = fetch_result['vectors'][vector_id]
            existing_metadata = vector_data.get('metadata', {})

            # Merge the existing metadata with the new metadata
            updated_metadata = {**existing_metadata, **new_metadata}

            # Ensure embedding is a list of floats
            embedding = [float(x) for x in embedding]

            # Update both embedding and metadata in Pinecone
            await asyncio.to_thread(
                self.index.upsert,
                vectors=[(vector_id, embedding, updated_metadata)]
            )

            logger.info(f"Successfully updated embedding and metadata for vector ID: {vector_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating vector in Pinecone for vector ID {vector_id}: {e}")
            logger.error("Full traceback:", exc_info=True)
            return False
    
    def update_memory_item_in_bigbird(self, memory_item: dict):
        # Similar adjustment as add_memory_item_to_bigbird to ensure correct text content is passed
        if isinstance(memory_item, MemoryItem):  # Adjust based on your class structure
            text_content = memory_item.content
        else:
            text_content = memory_item

        embedding, chunks = self.embedding_model.get_bigbird_embedding(text_content)
        for chunkembedding in embedding:
            self.bigbird_index.update(id=str(memory_item.id), values=chunkembedding)  # Ensure embedding is correctly formatted
            logger.info(f'Updated memory item with id {memory_item.id} in BigBird Pinecone index')

    def update_memory_item_in_embeddinglocal(self, memory_item: dict):
        # Similar adjustment as add_memory_item_to_bigbird to ensure correct text content is passed
        if isinstance(memory_item, MemoryItem):  # Adjust based on your class structure
            text_content = memory_item.content
        else:
            text_content = memory_item

        embedding = self.embedding_model.get_embeddinglocal(text_content)
        self.snowflake_index.update(id=str(memory_item.id), values=embedding.tolist())  # Ensure embedding is correctly formatted
        logger.info(f'Updated memory item with id {memory_item.id} in BigBird Pinecone index')

    def get_memory_item_from_bigbird(self, memory_item_id: str) -> Optional[Dict[str, Any]]:
        results = self.bigbird_index.fetch(ids=[memory_item_id])
        if results:
            return results[memory_item_id]
        else:
            return None
    
    async def get_memory_item_ids_from_conversation_history(
        self, 
        conversation_history: list[dict], 
        user_id: str
    ) -> List[str]:
        # Extract the content from each item in the conversation history
        conversation_content = [
            item['content'] for item in conversation_history 
            if 'content' in item
        ] if conversation_history else []
        
        embeddings = []
        for content in conversation_content:
            embedding, chunks = await self.embedding_model.get_sentence_embedding(content)
            if isinstance(embedding, list):
                embeddings.extend(embedding)
            else:
                embeddings.append(embedding)

        # Get user info for ACL filter
        user_instance = User.get(user_id)
        user_roles = user_instance.get_roles()
        user_workspace_ids = User.get_workspaces_for_user(user_id)
        
        logger.debug(f'user_roles {user_roles}')
        logger.info(f'user_workspace_ids {user_workspace_ids}')

        # Setup the ACL filter
        acl_filter = {
            "$or": [
                {"user_id": {"$eq": str(user_id)}},
                {"user_read_access": {"$in": [str(user_id)]}},
                {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}},
                {"role_read_access": {"$in": user_roles}},
            ]
        }

        # Query the Pinecone index asynchronously
        memory_item_ids = set()  # Use a set to avoid duplicates
        tasks = []
        
        for embedding in embeddings:
            tasks.append(
                self.index.query(
                    namespace="",
                    top_k=5,
                    include_values=True,
                    include_metadata=True,
                    vector=embedding,
                    filter=acl_filter
                )
            )

        # Wait for all queries to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            try:
                if isinstance(result, Exception):
                    logger.error(f"Error querying Pinecone: {result}")
                    continue
                    
                # Handle QueryResponse object properly
                matches = getattr(result, 'matches', None) if hasattr(result, 'matches') else result.get('matches')
                if matches and len(matches) > 0:
                    for match in matches:
                        memory_item_id = str(match.id if hasattr(match, 'id') else match['id'])
                        memory_item_ids.add(memory_item_id)
                        logger.info(f'Found memory item id {memory_item_id} for conversation content: {conversation_content[i]}')
            except Exception as e:
                logger.error(f"Error processing Pinecone result: {e}")
                continue

        return list(memory_item_ids)  # Convert set back to list before returning

    def add_memory_item(self, memory_item: MemoryItem, relationships_json: dict, sessionToken: str, user_id: str,  imageGenerationCategory=None, add_to_pinecone=True, workspace_id: str = None):
        import asyncio
        from services.user import User

        if memory_item and hasattr(memory_item, 'id'):
            self.memory_items[memory_item.id] = memory_item
        
        # If no workspace_id provided, try to get it from selected workspace follower
        if not workspace_id:
            workspace_id = User.get_selected_workspace_id(user_id, sessionToken)
            if workspace_id:
                logger.info(f"Using selected workspace ID: {workspace_id}")
                memory_item.metadata['workspace_id'] = workspace_id
            else:
                logger.warning("No workspace_id provided and no selected workspace found")
        
        # Define the categories for which we want to generate images
        IMAGE_GENERATION_CATEGORIES = {
            "narrative_element",
            "rpg_action",
            "object_description",
            "dream_or_fantasy",
            "art_idea",
            "historical_event",
            "biological_concept",
            "cultural_reference",
            "mood_or_emotion",
            "travel"
        }

        logger.info(f'imageGenerationCategory: {imageGenerationCategory}')

        # Initialize variables for the return values
        added_item_properties = None
        memory_item_obj = None

        if add_to_pinecone:
            # Add memory item without relationships first and wait for the result
            added_item_properties, memory_list = self.add_memory_item_without_relationships(sessionToken, memory_item)
            added_item_properties: List[ParseStoredMemory] = added_item_properties
            memory_list: List[MemoryItem] = memory_list

            if added_item_properties and memory_list:
                # Since we're now dealing with single items, get the first item
                added_item: ParseStoredMemory = added_item_properties[0]
                memory_item_obj: MemoryItem = memory_list[0]
                
                # Update the memory_item with objectId and createdAt from Parse response
                memory_item_obj.objectId = added_item.objectId
                memory_item_obj.createdAt = added_item.createdAt

                # Convert memory_item to a fully serializable dictionary
                memory_item_dict = memory_item_to_dict(memory_item_obj)

                # If the category is one for which we want to generate images
                if imageGenerationCategory and imageGenerationCategory in IMAGE_GENERATION_CATEGORIES:
                    logger.info(f'Image generation would be triggered here for category: {imageGenerationCategory}')
            
                if memory_item_obj:
                    # Create an async task for processing the memory item
                    async def process_memory_async():
                        await self.process_memory_item_async(
                            session_token=sessionToken,
                            memory_dict=memory_item_dict,
                            relationships_json=relationships_json,
                            workspace_id=workspace_id,
                            user_id=user_id
                        )

                    # Run the async task in a background thread
                    def run_async_task():
                        try:
                            # Create new event loop for this thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            # Run the async task
                            loop.run_until_complete(process_memory_async())
                            logger.info("Memory processing completed successfully")
                        except Exception as e:
                            logger.error(f"Error in memory processing: {e}")
                        finally:
                            loop.close()

                    # Start the background thread
                    import threading
                    thread = threading.Thread(target=run_async_task)
                    thread.start()
                    logger.info(f'Started async task to process memory item in background thread')

                    if memory_item_obj.context and len(memory_item_obj.context) > 0:
                        logger.info(f'Context for memory item exists: {memory_item_obj.context}')
                        self.update_memory_item_with_relationships(memory_item_obj, relationships_json, workspace_id, user_id)

        return [added_item_properties] if added_item_properties else []
    
    async def process_memory_item_async(self, session_token: str, memory_dict: dict, relationships_json: dict = None, workspace_id: str = None, user_id: str = None, user_workspace_ids: Optional[List[str]] = None) -> ProcessMemoryResponse:
        """
        Process a memory item asynchronously.
        
        Returns:
            Dict containing:
            - status_code (int): HTTP status code (200 for success, other codes for failures)
            - success (bool): Whether all steps completed successfully
            - data (Dict): Processing results and metrics
            - error (Optional[str]): Error message if any
        """
        try: 

            total_cost = 0
            success = True
            error = None

                        
            # Initialize ChatGPTCompletion
            logger.info("Starting process memory item")
            chat_gpt = ChatGPTCompletion(
                env.get('OPENAI_API_KEY'),
                env.get('OPENAI_ORG_ID'),
                env.get('LLM_MODEL'),
                env.get('LLM_LOCATION_CLOUD', default=True),
                env.get('EMBEDDING_MODEL_LOCAL')
            )

            # Constants for embedding costs
            BIGBIRD_EMBEDDING_COST = 0.0009043
            SENTENCE_BERT_COST = 0.0004521

            # Calculate memory item sizes - Use datetime_handler for serialization
            try:
                memory_item_text = json.dumps(memory_dict, default=self.datetime_handler)
                memory_item_storage_size = len(memory_item_text.encode('utf-8'))
                memory_item_token_size = chat_gpt.count_tokens(memory_item_text)
                
                logger.info(f'Successfully serialized memory_dict: {memory_item_text[:200]}...')  # Log first 200 chars
            except Exception as e:
                logger.error(f"Error serializing memory_dict: {e}")
                logger.error(f"memory_dict keys: {memory_dict.keys()}")
                logger.error(f"memory_dict types: {[(k, type(v)) for k, v in memory_dict.items()]}")
                raise

            logger.info(f'Initial memory item metrics:'
                    f'\n- Token size: {memory_item_token_size}'
                    f'\n- Storage size: {memory_item_storage_size} bytes')

            # Step 0: Fetch existing goals and use cases
            existing_goals = await get_user_goals_async(user_id, session_token)
            extracted_goals = extract_goal_titles(existing_goals)
            logger.debug(f'extracted_goals: {extracted_goals}')

            existing_use_cases = await get_user_usecases_async(user_id, session_token)
            logger.debug(f'existing_use_cases: {existing_use_cases}')
            extracted_use_cases = extract_usecases(existing_use_cases)
            logger.debug(f'extracted_use_cases: {extracted_use_cases}')

            # Get memory graph schema from structured outputs
            memory_graph_schema = self.get_memory_graph_schema()
            logger.debug(f'memory_graph_schema: {memory_graph_schema}')

            # Get simplified schema
            node_names, relationship_types = self.get_simplified_schema(memory_graph_schema)
            logger.debug(f'Node names: {node_names}')
            logger.debug(f'Relationship types: {relationship_types}')

            # Reconstruct memory_graph_schema in the expected format
            memory_graph_schema = {
                "nodes": node_names,
                "relationships": relationship_types
            }
            logger.debug(f'Reconstructed memory_graph_schema: {memory_graph_schema}')

            # Step 1: Generate usecase memory item
            usecase_response = await chat_gpt.generate_usecase_memory_item_async(
                memory_dict,
                memory_dict.get('context'),
                extracted_goals,
                extracted_use_cases
            )

            if not usecase_response:
                return {
                    "status_code": 500,
                    "success": False,
                    "error": "Failed to generate usecase memory item",
                    "data": None
                }

            if usecase_response:
                usecase_memory_item = usecase_response["data"]
                usecase_metrics = usecase_response["metrics"]
                logger.info(f'Generate usecase memory item: {usecase_memory_item}')
                logger.info(f'Usecase metrics - Input tokens: {usecase_metrics["usecase_token_count_input"]}, '
                        f'Output tokens: {usecase_metrics["usecase_token_count_output"]}, '
                        f'Total cost: ${usecase_metrics["usecase_total_cost"]:.4f}')

                # Process goals and use cases
                if usecase_memory_item.get('use_cases'):
                    new_use_cases = [uc for uc in usecase_memory_item["use_cases"] if uc["status"] == "new"]
                    if new_use_cases:
                        await add_list_of_usecases_async(user_id, session_token, new_use_cases)

                if usecase_memory_item.get('goals'):
                    new_goals = [goal for goal in usecase_memory_item["goals"] if goal["status"] == "new"]
                    if new_goals:
                        await add_list_of_goals_async(user_id, session_token, new_goals)

            # Step 2: Find related memories and build relationships and index memories in BigBird
            # Pass the current memory ID to exclude from related memories search
            related_memories_response = await chat_gpt.generate_related_memories_async(
                session_token, memory_graph_schema, memory_dict, user_id, 
                extracted_goals, extracted_use_cases, None,
                exclude_memory_id=memory_dict.get('id'),
                user_workspace_ids=user_workspace_ids
            )

            if not related_memories_response:
                return {
                    "status_code": 500,
                    "success": False,
                    "error": "Failed to generate related memories",
                    "data": None
                }
            
            if related_memories_response:

                related_memories: List[ParseStoredMemory] = related_memories_response["data"]
                generated_queries = related_memories_response["generated_queries"]
                related_memories_metrics = related_memories_response["metrics"]

                logger.info(f'Generated queries for finding related memories: {generated_queries}')
                logger.info(f'Generate list of memories to build relationships with: {related_memories}')
                logger.info(f'Related memories metrics - Input tokens: {related_memories_metrics["related_memories_token_count_input"]}, '
                        f'Output tokens: {related_memories_metrics["related_memories_token_count_output"]}, '
                        f'Total cost: ${related_memories_metrics["related_memories_total_cost"]:.6f}')

                # Trim and filter related memories
                trimmed_related_memories = self.trim_and_filter_related_memories(related_memories)

                # Create deterministic relationships with the top 3 related memories
                relationships_json = []
                for memory in related_memories[:3]:  # Limit to top 3 memories
                    relationship = {
                        "related_item_id": memory.memoryId,  # Updated to use proper field access
                        "relation_type": "RELATED_TO",
                        "metadata": {
                            "similarity_score": getattr(memory, 'score', 0),  # Use getattr for optional field
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                    relationships_json.append(relationship)

                logger.info(f'relationships_json: {relationships_json}')

                # When storing in memory_dict, ensure all datetime objects are converted
                if 'createdAt' in memory_dict:
                    memory_dict['createdAt'] = self.datetime_handler(memory_dict['createdAt'])
                
                if 'updatedAt' in memory_dict:
                    memory_dict['updatedAt'] = self.datetime_handler(memory_dict['updatedAt'])

                # When logging or serializing memory_dict
                memory_item_text = json.dumps(memory_dict, default=self.datetime_handler)
                memory_item_storage_size = len(memory_item_text.encode('utf-8'))

                # Index memories in BigBird
                await self.add_memory_item_to_bigbird(memory_dict, related_memories)

                # Process relationships between memories
                # Convert memory_dict to MemoryItem before passing to update_memory_item_with_relationships

                if relationships_json:
                    memory_item_obj = memory_item_from_dict(memory_dict)
                    relationship_result = await self.update_memory_item_with_relationships(
                        memory_item_obj,
                        relationships_json,
                        workspace_id,
                        user_id
                    )

                    if not relationship_result["success"]:
                        logger.warning(f"Failed to create some relationships: {relationship_result.get('error')}")
                        # Initialize empty relationship result if it failed
                        relationship_result = {
                            "success": False,
                            "relationships": [],
                            "error": relationship_result.get("error", "Failed to create relationships")
                        }
                else:
                    # Initialize empty relationship result if no relationships provided
                    relationship_result = {
                        "success": True,
                        "relationships": [],
                        "error": None
                    }

            
            # Step 3: Generate and store a memory graph for the memory item
            # Fetch user and workspace information
            try:
                user_info = await User.get_user_async(user_id)
                logger.info(f"user_info: {user_info}")
                company = await User.get_company_async(user_id, workspace_id, session_token)
                logger.info(f"company: {company}")
                
                # Add user and workspace info to memory_dict metadata
                if not memory_dict.get('metadata'):
                    memory_dict['metadata'] = {}
                
                memory_dict['metadata'].update({
                    'creator_name': user_info.get('name'),
                    'company': company
                })
                
                logger.info(f"Added creator and workspace info to memory: {memory_dict['metadata']}")
            except Exception as e:
                logger.error(f"Error fetching user/workspace info: {e}")

            try:
                schema_response = await chat_gpt.generate_memory_graph_schema_async(
                    memory_dict,
                    usecase_memory_item,  # We already have this from step 1
                    workspace_id,
                    trimmed_related_memories
                )

                if not schema_response:
                    return {
                        "status_code": 500,
                        "success": False,
                        "error": "Failed to generate memory graph schema",
                        "data": None
                    }

                # Get metrics directly from the response
                schema_metrics = schema_response.get("metrics", {
                    "schema_token_count_input": 0,
                    "schema_token_count_output": 0,
                    "schema_total_cost": 0,
                    "schema_total_tokens": 0
                })
                
                schema_total_cost = schema_metrics.get("schema_total_cost", 0)
        
                logger.info(f'Generated memory graph schema: {schema_response.get("data")}')
                logger.info(f'Schema metrics:'
                        f'\n- Input tokens: {schema_metrics.get("schema_token_count_input", 0)}'
                        f'\n- Output tokens: {schema_metrics.get("schema_token_count_output", 0)}'
                        f'\n- Total cost: ${schema_metrics.get("schema_total_cost", 0):.8f}'
                        f'\n- Total tokens: {schema_metrics.get("schema_total_tokens", 0)}')

                # Initialize metrics dictionary if it doesn't exist
                if 'metrics' not in memory_dict:
                    memory_dict['metrics'] = {'operation_costs': {}}
                    
                # Add schema generation cost to total metrics
                memory_dict['metrics']['operation_costs']['schema_generation'] = schema_total_cost

            except Exception as e:
                logger.error(f"Error generating memory graph schema: {e}")
                # Continue processing even if schema generation fails
                schema_data = None
                schema_metrics = {
                    "schema_token_count_input": 0,
                    "schema_token_count_output": 0,
                    "schema_total_cost": 0
                }
                schema_total_cost = 0

            # Fix the total cost calculation:
            total_cost = (
                (BIGBIRD_EMBEDDING_COST) +
                (SENTENCE_BERT_COST) +
                usecase_metrics["usecase_total_cost"] +
                related_memories_metrics["related_memories_total_cost"] +
                schema_metrics.get("schema_total_cost", 0)  # Use get() with default value
            )

            # Add complete metrics to memory_item
            memory_metrics = MemoryMetrics(
                total_cost=total_cost,
                token_size=memory_item_token_size,
                storage_size=memory_item_storage_size,
                operation_costs={
                    'usecase_generation': usecase_metrics["usecase_total_cost"],
                    'related_memories': related_memories_metrics["related_memories_total_cost"],
                    'schema_generation': schema_metrics["schema_total_cost"],
                    'bigbird_embedding': BIGBIRD_EMBEDDING_COST,
                    'sentence_bert': SENTENCE_BERT_COST
                }
            )
            memory_dict['metrics'] = memory_metrics

            # When logging the final metrics:
            logger.info(f'Memory item metrics:'
                    f'\n- Total cost: ${total_cost:.8f}'  # Changed from .6f to .8f
                    f'\n- Token size: {memory_item_token_size}'
                    f'\n- Storage size: {memory_item_storage_size} bytes'
                    f'\n- Operation costs breakdown:'
                    f'\n  * Usecase generation: ${usecase_metrics["usecase_total_cost"]:.8f}'  # Changed from .6f to .8f
                    f'\n  * Related memories: ${related_memories_metrics["related_memories_total_cost"]:.8f}'  # Changed from .6f to .8f
                    f'\n  * Schema generation: ${schema_metrics["schema_total_cost"]:.8f}'  # Changed from .6f to .8f
                    f'\n  * BigBird embedding: ${BIGBIRD_EMBEDDING_COST:.8f}'  # Changed from .6f to .8f
                    f'\n  * Sentence-BERT: ${SENTENCE_BERT_COST:.8f}')  # Changed from .6f to .8f

            # Update memory item with metrics
            if memory_dict.get('objectId'):
                logger.info(f'Updating memory item with metrics: {total_cost}')
                logger.info(f'memory_dict: {memory_dict}')
                
                await update_memory_item(session_token, memory_dict)
                
            elif memory_dict['objectId']:
                logger.info(f'Updating memory item with metrics: {total_cost}')
                logger.info(f'memory_dict: {memory_dict}')
                await update_memory_item(session_token, memory_dict)

            # When returning the final response, ensure all datetime objects are converted
            return {
                "status_code": 200,
                "success": True,
                "error": None,
                "data": {
                    "goal_usecases": usecase_memory_item,
                    "memory_graph": schema_response.get("data", {}),
                    "related_memories": [
                        {
                            **memory.model_dump(),
                            'createdAt': self.datetime_handler(memory.createdAt) if hasattr(memory, 'createdAt') else None,
                            'updatedAt': self.datetime_handler(memory.updatedAt) if hasattr(memory, 'updatedAt') else None
                        }
                        for memory in related_memories
                    ],
                    "related_memories_relationships": relationship_result.get("relationships", []) if relationships_json else [],
                    "metrics": memory_metrics
                }
            }
        except Exception as e:
            logger.error(f"Error in process_memory_item_async: {e}")
            return {
                "status_code": 500,
                "success": False,
                "error": str(e),
                "data": None
            }
           
    def datetime_handler(self, obj):
        """Handle serialization of datetime objects and other non-serializable types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            # Handle Pydantic models
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            # Handle other objects
            return str(obj)
        except Exception as e:
            logger.warning(f"Could not serialize object of type {type(obj)}: {e}")
            return None

    def get_simplified_schema(self, memory_graph_schema):
        """
        Extracts simplified schema information from the complex memory graph schema.
        Returns tuple of (node_names, relationship_types)
        """
        node_names = []
        relationship_types = []

        if memory_graph_schema:
            # Extract node names from schema
            nodes = memory_graph_schema.get('properties', {}).get('nodes', {}).get('items', {}).get('anyOf', [])
            if nodes:
                node_names = [
                    node.get('properties', {}).get('label', {}).get('enum', [])[0]
                    for node in nodes
                    if node.get('properties', {}).get('label', {}).get('enum')
                ]

            # Extract relationship types from schema
            relationships = memory_graph_schema.get('properties', {}).get('relationships', {}).get('items', {})
            if relationships:
                relationship_types = relationships.get('properties', {}).get('type', {}).get('enum', [])

        return node_names, relationship_types

    def trim_and_filter_related_memories(self, related_memories: List[ParseStoredMemory], max_length: int = 600) -> List[Dict[str, str]]:
        """
        Trims the content of related memories to a maximum length and filters to only include content.

        Args:
            related_memories (List[ParseStoredMemory]): List of ParseStoredMemory objects to process
            max_length (int, optional): Maximum length for content. Defaults to 300.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing trimmed memory data
        """
        if not related_memories:
            logger.info("No related memories provided for trimming")
            return []
            
        trimmed_memories: List[Dict[str, str]] = []
        for memory in related_memories:
            logger.debug(f"Processing memory for trimming: {memory.memoryId}")
            
            # Access content directly from ParseStoredMemory object
            content = memory.content
            if not content:
                logger.info(f"Memory {memory.memoryId} has no content, skipping")
                continue
                
            trimmed_memory = {
                'id': memory.memoryId,
                'content': (content[:max_length] + '...') if len(content) > max_length else content
            }
            
            trimmed_memories.append(trimmed_memory)
            logger.debug(f"Added trimmed memory: {trimmed_memory['id']}")

        logger.info(f'Trimmed memories: {trimmed_memories}')
        logger.info(f"Trimmed {len(related_memories)} memories to {len(trimmed_memories)} with content")
        return trimmed_memories
    
    def flatten_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def validate_metadata(self, metadata):
        """Validate and flatten metadata for Neo4j storage."""
        if not metadata:
            logger.warning("Empty metadata received")
            return {}

        # Log the input metadata
        logger.info(f"Validating metadata: {json.dumps(metadata, indent=2)}")

        # Ensure all keys and values are of valid types
        validated_metadata = {}
        for key, value in metadata.items():
            # Skip None values
            if value is None or value == "None":
                continue
                
            if not isinstance(key, str):
                logger.warning(f"Skipping invalid key type: {key}")
                continue
                
            # Handle different value types
            if isinstance(value, (str, int, float, bool)):
                validated_metadata[key] = value
            elif isinstance(value, list):
                # Keep lists as lists for Neo4j
                validated_metadata[key] = value if value else []
            else:
                # Convert other types to strings
                validated_metadata[key] = str(value)

        logger.info(f"Validated metadata: {json.dumps(validated_metadata, indent=2)}")
        return validated_metadata

    async def add_memory_item_to_neo4j(
        self, 
        memory_item: MemoryItem, 
        memoryChunkIds: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, storing memory item in fallback storage")
                #self.fallback_storage[str(memory_item.id)] = memory_item.dict()
                return {"id": str(memory_item.id)}
            
            memory_id = str(memory_item.id)
            chunk_ids = memoryChunkIds if memoryChunkIds else [memory_id]
            
            # Check if node exists
            existing_id = await self._node_exists(
                node_id=memory_id,
                node_type=NodeLabel.Memory,
                node_content=memory_item.content
            )
            
            if existing_id:
                logger.info(f"Memory node already exists with ID: {existing_id}")
                return {"id": existing_id}

            # Convert MemoryItem to Node
            memory_node = memory_item_to_node(memory_item, chunk_ids)
            
            # Use existing _create_node method
            #driver = await self.async_neo_conn.get_driver()
            #async with driver.session() as session:
            async with self.async_neo_conn.get_session() as session:
                result = await self._create_node(
                    session=session,
                    node=memory_node,
                    common_metadata={}  # Already included in node properties
                )
                return result

        except Exception as e:
            logger.error(f"Error adding memory node to Neo4j: {e}", exc_info=True)
            raise

    async def update_memory_item_in_neo4j(
        self, 
        memory_item_dict: dict, 
        memory_type: Optional[str] = None, 
        memoryChunkIds: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:        
        """
        Updates an existing memory item's metadata in Neo4j.
        Excludes the 'content' field from updates.

        Parameters:
            memory_item (dict): A dictionary representing the memory item.
            memory_type (str, optional): The new type to set for the memory item. Defaults to None.
            memoryChunkIds (list, optional): List of Pinecone chunk IDs to update. Defaults to None.

        Returns:
            dict: The properties of the updated node, or None if no update occurred.
        """
        logger.info(f"Updating memory item in Neo4j with ID: {memory_item_dict['id']}")

        # Define the metadata fields to include
        metadata_fields = [
            "id",
            "user_id",
            "pageId",
            "hierarchical_structures",
            "type",
            "title",
            "topics",
            "conversationId",
            "prompt",
            "imageURL",
            "sourceType",
            "sourceUrl",
            "workspace_id",
            "user_id",
            "user_read_access",
            "user_write_access",
            "workspace_read_access",
            "workspace_write_access",
            "role_read_access",
            "role_write_access",
            "content"
        ]

        # Define key mappings: Parse Server keys to Neo4j keys
        key_mappings = {
            'emojiTags': 'emoji_tags',
            'emoji tags': 'emoji_tags',
            'emotionTags': 'emotion_tags',
            'emotion tags': 'emoji_tags',
            'hierarchicalStructures': 'hierarchical_structures',
            'hierarchical structures': 'hierarchical_structures'
        }


        try:
            # Ensure Neo4j connection is initialized
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, storing memory item in fallback storage")
                #self.fallback_storage[memory_item_dict['id']] = memory_item_dict
                return memory_item_dict
            #driver = await self.async_neo_conn.get_driver()
            logger.info("Neo4j driver initialized successfully")

            # Extract relevant metadata
            relevant_metadata = {field: memory_item_dict.get(field) for field in metadata_fields}
            logger.info(f"Extracted relevant metadata: {relevant_metadata}")

            # Prepare the properties for the updated node
            properties = {}
            
            # Only add properties that exist in memory_item_dict
            possible_properties = [
                'content',
                'title',
                'user_read_access',
                'user_write_access',
                'workspace_read_access',
                'workspace_write_access',
                'role_read_access',
                'role_write_access',
                'sourceUrl',
                'memoryChunkIds',
                "type"
            ]

            for prop in possible_properties:
                if prop in memory_item_dict and memory_item_dict[prop] is not None:
                    properties[prop] = memory_item_dict[prop]

            if memory_type:
                properties["type"] = memory_type
            elif "type" in memory_item_dict:
                properties["type"] = memory_item_dict["type"]

            # Process metadata fields with key mapping
            for field in metadata_fields:
                # Try both original and mapped keys
                value = None
                for parse_key, neo_key in key_mappings.items():
                    if neo_key == field and parse_key in memory_item_dict:
                        value = memory_item_dict[parse_key]
                        break
                if value is None:
                    value = memory_item_dict.get(field)

                if value is not None and field != 'id':
                    # Handle specific field conversions
                    if field in ['emoji_tags', 'emotion_tags']:
                        if isinstance(value, str):
                            properties[field] = [tag.strip() for tag in value.split(',') if tag.strip()]
                        elif isinstance(value, list):
                            properties[field] = value
                    else:
                        properties[field] = value

            # Update memoryChunkIds if provided
            if memoryChunkIds is not None:
                properties["memoryChunkIds"] = memoryChunkIds

            logger.info(f"Properties to update: {properties}")

            #async with driver.session() as session:
            async with self.async_neo_conn.get_session() as session:
                # First, check if the node exists
                check_query = """
                    MATCH (n:Memory {id: $id})
                    RETURN n
                """
                check_result = await session.run(check_query, id=str(memory_item_dict['id']))
                check_record = await check_result.single()
                
                if not check_record:
                    logger.error(f"Node with ID {memory_item_dict['id']} not found in Neo4j")
                    return None

                logger.info("Found existing node in Neo4j, proceeding with update")
                logger.info(f"Existing node properties: {dict(check_record['n'])}")

                logger.info(f"Properties to update: {properties}")

                # Prepare the Cypher query - using direct SET
                query = """
                    MATCH (n:Memory {id: $id})
                    SET n += $properties
                    RETURN n
                """

                logger.info(f"Executing Cypher query: {query}")
                logger.info(f"Query parameters - id: {str(memory_item_dict['id'])}")
                logger.info(f"Query parameters - properties: {properties}")

                try:
                    # Execute the query with merged properties
                    result = await session.run(
                        query,
                        id=str(memory_item_dict['id']),
                        properties=properties   
                    )
                    record = await result.single()
                    logger.info(f"Record type: {type(record)}")
                    logger.info(f"Record keys: {record.keys() if record else 'No record'}")
                    logger.info(f"Record values: {record.values() if record else 'No record'}")
                    
                    if record:
                        # Try to get node data, handling different record structures
                        node_data = None
                        if 'n' in record:
                            node_data = dict(record['n'])
                        elif len(record) > 0:
                            # If 'n' isn't explicitly keyed but we have data
                            node_data = dict(record[0])
                        
                        if node_data:
                            logger.info(f"Successfully updated node properties: {node_data}")
                            return node_data
                        else:
                            # We got a record but couldn't extract node data
                            logger.warning(f"Update succeeded but couldn't extract node data from record: {record}")
                            # Return the properties we sent since we know they were applied
                            return properties
                    else:
                        logger.error(f"No record returned for memory item with ID {memory_item_dict['id']}")
                        return None
                except Exception as query_error:
                    logger.error(f"Error executing Neo4j query: {query_error}", exc_info=True)
                    raise

        except Exception as e:
            logger.error(f"Error updating memory node in Neo4j: {e}", exc_info=True)
            raise

    async def lookup_memory_by_client_msg_id(self, client_msg_id: str) -> Optional[str]:
        """
        Asynchronously lookup a memory by its client message ID.
        
        Args:
            client_msg_id (str): The client message ID to look up
            
        Returns:
            Optional[str]: The memory ID if found, None otherwise
        """
        try:
            # Ensure Neo4j connection is initialized
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, cannot lookup memory by client_msg_id")
                return None
            #driver = await self.async_neo_conn.get_driver()
            
            #async with driver.session() as session:
            async with self.async_neo_conn.get_session() as session:
                result = await session.run(
                    "MATCH (m:Memory {client_msg_id: $client_msg_id}) RETURN m.id AS id",
                    client_msg_id=client_msg_id
                )
                record = await result.single()
                if record:
                    return record['id']
                return None
            
        except Exception as e:
            logger.error(f"Error looking up memory by client_msg_id: {str(e)}")
            return None


    
    async def find_related_memory_items_async(
        self, 
        session_token: str, 
        query: str, 
        context: str, 
        user_id: str, 
        chat_gpt: "ChatGPTCompletion", 
        metadata, 
        relation_type: str = None, 
        project_id: str = None, 
        skip_neo: bool = True, 
        exclude_memory_id: str = None,
        neo_session: Optional[AsyncSession] = None,
        user_workspace_ids: Optional[List[str]] = None,
        reranking_config: Optional[Dict[str, Any]] = None
    ) -> RelatedMemoryResult:
        """
        Find related memory items using various sources (Pinecone, BigBird, Neo4j).
        Returns structured results including memory items and neo nodes.
        """
        fetch_start = time.time()  # Initialize at the start of the method
        start_time = time.time()
        neoQuery = None
        mem_source_dict = {}
        if reranking_config:
            reranking_enabled = reranking_config.get('reranking_enabled', False)
            reranking_model = reranking_config.get('reranking_model', 'gpt-4o-mini-2024-07-18')
        else:
            reranking_enabled = False
            reranking_model = None
        result = RelatedMemoryResult(
            memory_items=[],
            neo_nodes=[],
            neo_context=None,
            neo_query=None,  # Added this field
            memory_source_info=MemorySourceInfo(memory_id_source_location=[])  # Initialize with empty list
        )

        # Only ensure connection if we don't have a session
        if not neo_session and not skip_neo:
            start_neo_time = time.time()
            await self.ensure_async_connection()
            neo_time = time.time() - start_neo_time
            logger.warning(f"Neo4j connection ensure took {neo_time:.2f}s (fallback_mode: {self.async_neo_conn.fallback_mode})")
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, skipping Neo4j query")
                return result

        # Only use the query for better accuracy
        query_context_combined = query
        
        # Get user info (roles and workspaces) - convert to async calls
        start_user_time = time.time()
        user_instance = User.get(user_id)
        logger.info(f'user_instance find_related_memory_items_async {user_instance}')

        # Run both queries in parallel
        if user_instance:
            user_roles, user_workspace_ids = await asyncio.gather(
                user_instance.get_roles_async(),
                User.get_workspaces_for_user_async(user_id) if user_workspace_ids is None else asyncio.sleep(0, user_workspace_ids)
            )
        else:
            user_roles = []
            user_workspace_ids = []
            
        user_time = time.time() - start_user_time
        logger.warning(f"User info retrieval took {user_time:.2f}s (roles: {len(user_roles)}, workspaces: {len(user_workspace_ids) if user_workspace_ids else 0})")
        logger.warning(f"User info retrieval took {user_time:.2f}s (roles: {len(user_roles)}, workspaces: {len(user_workspace_ids) if user_workspace_ids else 0})")
        
        # Setup the ACL filter using the working structure
        #acl_filter = {
        #    "$or": [
        #        {"user_id": {"$eq": str(user_id)}},
        #        {"user_read_access": {"$in": [str(user_id)]}},
        #        {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}},
        #        {"role_read_access": {"$in": user_roles}},
        #    ]
        #}

        # Setup the ACL filter dynamically based on available data
        acl_conditions = [
            {"user_id": {"$eq": str(user_id)}},
            {"user_read_access": {"$in": [str(user_id)]}}
        ]
        
        # Only add workspace conditions if we have workspace IDs
        if user_workspace_ids:
            acl_conditions.append(
                {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}}
            )
            
        # Only add role conditions if we have roles
        if user_roles:
            acl_conditions.append(
                {"role_read_access": {"$in": user_roles}}
            )
            
        acl_filter = {"$or": acl_conditions}
        
        logger.info(f"ACL filter constructed with: user_id, {len(user_workspace_ids) if user_workspace_ids else 0} workspaces, {len(user_roles) if user_roles else 0} roles")
        
        logger.debug(f"Pinecone ACL Filter: {acl_filter}")

        embedding_start = time.time()
        # Run embeddings in parallel using asyncio.gather
        query_context_embedding, bigbird_embedding = await asyncio.gather(
            self.embedding_model.get_sentence_embedding(query_context_combined),
            self.embedding_model.get_bigbird_embedding(query_context_combined)
        )
        
        embedding_time = time.time() - embedding_start
        logger.warning(f"Embedding generation took find_related_memory_items_async {embedding_time:.4f} seconds")

        # Run Pinecone and BigBird queries in parallel using asyncio.gather
        search_start = time.time()
        similar_embeddings_results, bigbird_memory_info = await asyncio.gather(
            self.get_pinecone_related_memories_async(
                query_context_embedding[0],
                acl_filter,  # Use the direct dictionary format
                top_k=20
            ),
            self.get_bigbird_related_memories_async(
                bigbird_embedding[0],
                acl_filter,  # Use the direct dictionary format
                user_id,
                top_k=15
            )
        )
        
        search_time = time.time() - search_start
        logger.warning(f"Vector search took {search_time:.4f} seconds")

        # Process results efficiently using list comprehensions
        memory_item_ids = [match['id'] for match in similar_embeddings_results.get('matches', [])]
        bigbird_memory_ids = [memory_info['id'] for memory_info in bigbird_memory_info]

        neo_memory_ids = []
        similar_memory_items = []

        neo_start = time.time()

        async with contextlib.AsyncExitStack() as stack:
            # Use existing session or create new one
            session = neo_session or await stack.enter_async_context(self.async_neo_conn.get_session())

            if not skip_neo:
                

                try:
                    memory_nodes, other_nodes, _, text_context = await self.query_neo4j_with_user_query_async(
                        session_token, 
                        query_context_combined, 
                        memory_item_ids, 
                        acl_filter,
                        user_id, 
                        chat_gpt, 
                        project_id, 
                        top_k=50,
                        neo_session=session
                    )
                    result.neo_nodes = other_nodes
                    result.neo_context = text_context
                    
                    # Add memory nodes to memory_items if they exist
                    if memory_nodes:
                        try:
                            neo_memory_items = await self.fetch_memory_items_from_sources_async(
                                session_token,
                                [node.id for node in memory_nodes],
                                user_id,
                                session
                            )
                            result.memory_items.extend(neo_memory_items)
                            neo_memory_ids = [node.id for node in memory_nodes]
                        except Exception as e:
                            logger.warning(f"Error fetching Neo4j memory items: {e}. Continuing without Neo4j memory items.")
                            # Keep neo_memory_ids empty
                            pass

                except Exception as e:
                    logger.warning(f"Error querying Neo4j: {e}. Continuing without Neo4j results.")
                    # Reset Neo4j related fields to empty/None values
                    result.neo_nodes = []
                    result.neo_context = None
                    result.neo_query = None
                    # Keep neo_memory_ids empty
                    pass

                neo_time = time.time() - neo_start
                logger.warning(f"Neo4j query took {neo_time:.4f} seconds")


            # Process memory IDs to get base IDs without chunk numbers
            def get_base_id(memory_id):
                return memory_id.split('_')[0] if '_' in memory_id else memory_id
            
            # Get base IDs for each source
            memory_base_ids = [get_base_id(mid) for mid in memory_item_ids]
            bigbird_base_ids = [get_base_id(mid) for mid in bigbird_memory_ids]
            neo_base_ids = [get_base_id(mid) for mid in neo_memory_ids]
            
            # Combine IDs efficiently using set operations
            combined_memory_item_ids = list(set(memory_item_ids + bigbird_base_ids + neo_base_ids))
            logger.info(f'combined_memory_item_ids len {len(combined_memory_item_ids)}')
            
            # Update the memory source info creation
            memory_id_source_locations = [
                MemoryIDSourceLocation(
                    memory_id=item,
                    source_location=MemorySourceLocation(
                        in_pinecone=item in set(memory_item_ids),
                        in_bigbird=item in set(bigbird_memory_ids),
                        in_neo=item in set(neo_memory_ids)
                    )
                )
                for item in combined_memory_item_ids
            ]
            
            result.memory_source_info = MemorySourceInfo(
                memory_id_source_location=memory_id_source_locations
            )
            # Fetch memory items if we have results
            if combined_memory_item_ids:
                fetch_start = time.time()
                try:
                    similar_memory_items: List[ParseStoredMemory] = await self.fetch_memory_items_from_sources_async(
                        session_token, 
                        combined_memory_item_ids, 
                        user_id,
                        neo_session=session
                    )

                    # Filter out excluded memory ID if specified and update result
                    if similar_memory_items:
                        result.memory_items = [
                            item for item in similar_memory_items 
                            if not exclude_memory_id or item.memoryId != exclude_memory_id
                        ]

                    logger.info(f"Filtered memory items count: {len(similar_memory_items)}")
                    logger.debug(f"First memory item type: {type(similar_memory_items[0]) if similar_memory_items else 'No items'}")
                except Exception as e:
                    logger.error(f"Error fetching combined memory items: {e}")
            
            # Rerank results if enabled
            if reranking_enabled and result.memory_items:
                logger.info("Reranking memory items based on relevance scores")
                try:
                    # Get relevance scores for each memory item
                    scores = []
                    for memory_item in result.memory_items:
                        messages = [
                            {"role": "system", "content": "You are an assistant that helps evaluate search results based on their relevance to a given query. Please provide only a relevance score between 1 and 10."},
                            {"role": "user", "content": f"Here is the query: '{query}'.\n\nHere is the result: '{memory_item.content}'.\n\nEvaluate how relevant the result is to the query on a scale of 1 to 10, where 1 means not relevant at all and 10 means highly relevant. Please provide only the relevance score as a number between 1 and 10."}
                        ]
                        
                        response = await chat_gpt._create_completion_async(
                            model=reranking_model,
                            messages=messages,
                            max_tokens=10,
                            temperature=0.3
                        )
                        
                        try:
                            score = float(response.choices[0].message.content.strip())
                            scores.append((score, memory_item))
                        except ValueError:
                            logger.warning(f"Could not parse relevance score from response: {response.choices[0].message.content}")
                            scores.append((0, memory_item))
                            
                    # Sort memory items by score in descending order
                    scores.sort(reverse=True, key=lambda x: x[0])
                    result.memory_items = [item for _, item in scores]
                    
                    logger.info("Successfully reranked memory items")
                except Exception as e:
                    logger.error(f"Error during reranking: {e}")
                    # Keep original order if reranking fails
            result.log_summary()

            total_fetch_time = time.time() - fetch_start
            total_time = time.time() - start_time
            logger.warning(f"Total total_fetch_time execution took {total_fetch_time:.4f} seconds")
            logger.warning(f"Total find_related_memory_items execution took {total_time:.4f} seconds")
            logger.warning(f'len memory_items {len(result.memory_items)}')
            
            return result
    
            
    def fetch_parse_server(self, session_token: str, memory_item_ids: List[str], chunk_base_ids: List[str]) -> Dict[str, Any]:
        """
        Worker function to fetch memory items from Parse Server.
        Args:
            session_token (str): Authentication token for Parse Server.
            memory_item_ids (List[str]): List of memory item IDs to fetch.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - results: List of memory items fetched from Parse Server
                - missing_memory_ids: List of memory IDs that weren't found
        """
        try:
            response = retrieve_multiple_memory_items(session_token, memory_item_ids, chunk_base_ids)
            logger.info(f"Retrieved {len(response['results'])} memory items from Parse Server.")
            logger.info(f"Missing {len(response['missing_memory_ids'])} memory items.")
            return response  # Already contains 'results' and 'missing_memory_ids'
        except Exception as e:
            logger.error(f"Error fetching from Parse Server: {e}")
            return {'results': [], 'missing_memory_ids': memory_item_ids}
        
    async def query_neo4j_async(self, query: str, params: Dict[str, Any], neo_session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """
        Async worker function to query Neo4j.
        Each connection creates its own session.
        
        Args:
            query (str): The Cypher query to execute.
            params (Dict[str, Any]): Parameters for the Cypher query.
            neo_session (Optional[AsyncSession]): Existing Neo4j session to reuse.
        
        Returns:
            List[Dict[str, Any]]: A list of memory items fetched from Neo4j.
        """
        try:
        
            if not neo_session:
                # Only ensure connection if we don't have a session
                await self.ensure_async_connection()
                if self.async_neo_conn.fallback_mode:
                    logger.warning("Neo4j in fallback mode, returning empty results")
                    return []

            async with contextlib.AsyncExitStack() as stack:
                # Use existing session or create new one
                session = neo_session or await stack.enter_async_context(self.async_neo_conn.get_session())

                result = await session.run(query, params)
                records = []
                async for record in result:
                    records.append(record)
                await result.consume()
                
                # Track unique items by both ID and content
                unique_items = {}
                content_map = {}  # Map content to IDs for duplicate detection
                
                for record in records:
                    node = record['a']
                    node_dict = dict(node)
                    node_id = node_dict.get('id')
                    content = node_dict.get('content')
                    
                    if not node_id:
                        logger.warning(f"Memory ID property missing from node: {node_dict}")
                        continue
                    
                    # Check for content-based duplicates if content exists
                    if content:
                        if content in content_map:
                            # Content already exists, log duplicate
                            logger.debug(f"Duplicate content found for IDs: {node_id} and {content_map[content]}")
                            # Keep the record with the earlier ID (assuming string comparison)
                            if node_id < content_map[content]:
                                # Remove old entry
                                old_id = content_map[content]
                                unique_items.pop(old_id, None)
                                # Add new entry
                                content_map[content] = node_id
                                unique_items[node_id] = node_dict
                        else:
                            # New content
                            content_map[content] = node_id
                            unique_items[node_id] = node_dict
                    else:
                        # No content, check ID-based duplicate
                        if node_id in unique_items:
                            # Keep entry with content if it exists
                            if not unique_items[node_id].get('content'):
                                unique_items[node_id] = node_dict
                        else:
                            unique_items[node_id] = node_dict
                
                memory_items = list(unique_items.values())
                
                logger.info(f"Processed {len(records)} Neo4j records into {len(memory_items)} unique items")
                logger.info(f"Removed {len(records) - len(memory_items)} duplicates")
                
                return memory_items
                
        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
            return []

    async def fetch_parse_server_async(
        self, 
        session_token: str, 
        memory_item_ids: List[str], 
        chunk_base_ids: List[str], 
        memory_class: str = "Memory"
    ) -> MemoryRetrievalResult:
        """
        Async worker function to fetch memory items from Parse Server.
        Args:
            session_token (str): Authentication token for Parse Server.
            memory_item_ids (List[str]): List of memory item IDs to fetch.
            chunk_base_ids (List[str]): List of base IDs without chunk numbers.
        
        Returns:
            MemoryRetrievalResult: A dictionary containing:
                - 'results': List[ParseStoredMemory] - List of memory items with user objects
                - 'missing_memory_ids': List[str] - List of memory IDs that weren't found
        """
        try:
            # Assuming retrieve_multiple_memory_items is also async
            response: MemoryRetrievalResult = await retrieve_memory_items_with_users_async(
                session_token, 
                memory_item_ids, 
                chunk_base_ids, 
                memory_class
            )

            # Access the TypedDict fields properly
            logger.info(f"Retrieved {len(response.get('results', []))} memory items from Parse Server.")
            logger.info(f"Missing {len(response.get('missing_memory_ids', []))} memory items.")
            return response  # Already contains 'results' and 'missing_memory_ids'
        except Exception as e:
            logger.error(f"Error fetching from Parse Server: {e}")
            return {'results': [], 'missing_memory_ids': memory_item_ids}
            
    def normalize_and_merge_memory_items(
        self, 
        similar_memory_items_neo: List[Dict[str, Any]], 
        memory_items_parse: List[ParseStoredMemory]
    ) -> List[ParseStoredMemory]:
        """
        Normalize and merge memory items from Neo4j and Parse Server.
        Always prefer Parse Server items as they are the source of truth.
        Returns Parse Server items without metadata.

        Args:
            similar_memory_items_neo: List of memory items from Neo4j
            memory_items_parse: List of ParseStoredMemory items from Parse Server

        Returns:
            List[ParseStoredMemory]: List of normalized Parse Server memory items without metadata
        """
        logger.info('Starting normalization and merging of memory items.')
        
        # Create sets for tracking IDs
        neo_ids: Set[str] = set()
        parse_ids: Set[str] = set()

        # Track Neo4j items for logging purposes
        logger.info(f"Neo4j fetched {len(similar_memory_items_neo)} items.")
        for item in similar_memory_items_neo:
            memory_id = item.get('id')
            if memory_id:
                neo_ids.add(memory_id)
                logger.debug(f"Neo4j Memory item - Memory ID: {memory_id}")
            else:
                logger.warning(f"Memory ID property missing from Neo4j item: {item}")

        # Process Parse Server items (source of truth)
        unique_parse_items: Dict[str, ParseStoredMemory] = {}
        logger.info(f"Parse Server fetched {len(memory_items_parse)} items.")
        
        for item in memory_items_parse:
            memory_id = item.memoryId
            if memory_id:
                parse_ids.add(memory_id)
                # Create a copy without metadata using the model method
                unique_parse_items[memory_id] = item.without_metadata()
                logger.debug(f"Parse Memory item - Memory ID: {memory_id}, ObjectId: {item.objectId}")

        # Log any discrepancies between Neo4j and Parse Server
        missing_in_parse = neo_ids - parse_ids
        missing_in_neo = parse_ids - neo_ids

        if missing_in_parse:
            logger.warning(f'Memory IDs in Neo4j but missing in Parse: {list(missing_in_parse)}')
        if missing_in_neo:
            logger.debug(f'Memory IDs in Parse but missing in Neo4j: {list(missing_in_neo)}')

        normalized_items = list(unique_parse_items.values())
        logger.info(f'Final normalized items count: {len(normalized_items)}')
        
        return normalized_items

    def handle_missing_parse_items(
        self, 
        similar_memory_items_neo: List[Dict[str, Any]], 
        normalized_memory_items: List[Dict[str, Any]], 
        session_token: str, 
        missing_memory_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Handle memory items missing in Parse Server by adding them from Neo4j.
        
        Args:
            similar_memory_items_neo (List[Dict[str, Any]]): Memory items from Neo4j.
            normalized_memory_items (List[Dict[str, Any]]): Normalized memory items from Parse Server.
            session_token (str): Authentication token for Parse Server.
            missing_memory_ids (List[str]): List of memory item IDs missing in Parse Server.
        
        Returns:
            Dict[str, Any]: A dictionary mapping memory IDs to their respective memory items.
        """
        unique_memory_items: Dict[str, Any] = {}

        # Create a mapping of Parse Server items by memoryId
        parse_items_map = {
            item.get('memoryId'): item 
            for item in normalized_memory_items 
            if 'objectId' in item and item.get('memoryId') is not None
        }
        
        # Create a mapping of Neo4j items by memoryId for quick lookup
        neo_items_map = {
            item.get('id'): item 
            for item in similar_memory_items_neo 
            if item.get('id') and item.get('content')
        }

        # Process only the missing_memory_ids
        for memory_id in missing_memory_ids:
            item = neo_items_map.get(memory_id)
            if not item:
                logger.warning(f"Memory item {memory_id} not found in Neo4j or lacks 'content'. Skipping.")
                continue

            try:
                memory_item_obj = memory_item_from_dict(item)

                # Ensure 'content' exists for legacy items
                if not memory_item_obj.content:
                    logger.error(f"Legacy memory item {memory_id} lacks 'content'. Skipping.")
                    continue

                neo_item_user_id = item.get('user_id')

                # Store in Parse Server
                parse_response = store_generic_memory_item(neo_item_user_id, session_token, memory_item_obj)
                parse_object = retrieve_memory_item_with_user(session_token, memory_id)
                
                if parse_object and 'objectId' in parse_object:
                    unique_memory_items[memory_id] = parse_object
                    logger.info(f"Added memory item {memory_id} to Parse Server with objectId {parse_object['objectId']}.")

                    # Optionally remove content and context from Neo4j after successful addition
                    #try:
                    #    with self.neo_conn.session() as session:
                    #        deletion_result = session.run(
                    #            """
                    #            MATCH (m:Memory {id: $memory_id})
                    #            SET m.content = '', m.context = ''
                    #            RETURN count(m) as updatedCount
                    #            """,
                    #            memory_id=memory_id
                    #            ).single()

                    #            updated_count = deletion_result.get('updatedCount', 0)
                    #            if updated_count > 0:
                    #                logger.info(f"Content and context removed from Neo4j for memory_id {memory_id}.")
                    #else:
                    #    logger.warning(f"No Neo4j nodes updated for memory_id {memory_id}.")
                    #except Exception as e:
                    #    logger.error(f"Error removing content and context from Neo4j for memory_id {memory_id}: {e}")
                
                else:
                    logger.error(f"Failed to retrieve memory item {memory_id} from Parse Server after addition.")
            except Exception as e:
                logger.error(f"Error processing memory item {memory_id}: {e}")
                continue

        return unique_memory_items

    
    async def fetch_memory_items_from_sources_async(
        self, 
        session_token: str, 
        memory_item_ids: List[str], 
        user_id: str,
        neo_session: Optional[AsyncSession] = None
    ) -> List[ParseStoredMemory]:
        """
        Fetch memory items from Neo4j and Parse Server concurrently, normalize and merge them.
        
        Args:
            session_token (str): Authentication token for Parse Server.
            memory_item_ids (List[str]): List of memory item IDs to fetch (may include chunked IDs with _#).
            user_id (str): ID of the user requesting the memory items.
        
        Returns:
            List[ParseStoredMemory]: A list of final memory items after processing.
        """
        if not neo_session:
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, cannot fetch memory items")
                return []
            
        # Process memory IDs to handle both legacy and chunked formats
        base_memory_ids = set()  # Use set to avoid duplicates
        chunk_id_mapping = {}  # Map to track original chunked IDs
        
        for memory_id in memory_item_ids:
            # Check if the ID has a chunk suffix (_#)
            if '_' in memory_id:
                base_id = memory_id.rsplit('_', 1)[0]  # Get the base ID without chunk number
                base_memory_ids.add(base_id)
                chunk_id_mapping[base_id] = memory_id  # Store mapping of base ID to chunked ID
            else:
                base_memory_ids.add(memory_id)
            logger.info(f'Base memory IDs: {base_memory_ids}')
        processed_memory_ids = list(base_memory_ids)
        logger.debug(f'Processed memory IDs: {processed_memory_ids}')
        logger.info(f'Chunk ID mapping: {chunk_id_mapping}')

        # Define the Neo4j batch query
        neo_query = '''
            MATCH (a:Memory) 
            WHERE a.id IN $ids 
            OR (a.memoryChunkIds IS NOT NULL AND ANY(chunk_id IN a.memoryChunkIds WHERE chunk_id IN $memoryChunkIds))
            OR (a.id IN $chunk_base_ids)
            RETURN a
        '''
        neo_params = {
            'ids': processed_memory_ids,
            'memoryChunkIds': memory_item_ids,
            'chunk_base_ids': processed_memory_ids
        }

        # Combine original and processed IDs for Parse Server query
        all_memory_ids = list(set(memory_item_ids + processed_memory_ids))

        async with contextlib.AsyncExitStack() as stack:
            # Use existing session or create new one
            session = neo_session or await stack.enter_async_context(self.async_neo_conn.get_session())

            try:
        
                # Execute Neo4j and Parse Server queries concurrently
                memory_class: str = "Memory"
                similar_memory_items_neo, parse_response = await asyncio.gather(
                    self.query_neo4j_async(neo_query, neo_params, session),
                    self.fetch_parse_server_async(session_token, all_memory_ids, processed_memory_ids, memory_class),
                    return_exceptions=True  # This allows gather to complete even if one call fails
                )

                # Handle potential exceptions from gather results
                if isinstance(similar_memory_items_neo, Exception):
                    logger.error(f"Neo4j query failed: {similar_memory_items_neo}")
                    similar_memory_items_neo = []
                else:
                    logger.info(f'Neo4j fetched {len(similar_memory_items_neo)} items.')

                if isinstance(parse_response, Exception):
                    logger.error(f"Parse Server query failed: {parse_response}")
                    memory_items_parse = []
                else:
                    memory_items_parse = parse_response['results']
                    missing_memory_ids = parse_response['missing_memory_ids']
                    logger.info(f'Parse Server fetched {len(memory_items_parse)} items.')
                    logger.info(f'Parse Server missing {len(missing_memory_ids)} items.')

                # If both sources failed, return empty list
                if not similar_memory_items_neo and not memory_items_parse:
                    logger.error("Both Neo4j and Parse Server queries failed")
                    return []
                
                # Extract Parse Server results
                parse_response: MemoryRetrievalResult = parse_response
                memory_items_parse: List[ParseStoredMemory] = parse_response['results']
                missing_memory_ids: List[str] = parse_response['missing_memory_ids']

                logger.info(f'Neo4j fetched {len(similar_memory_items_neo)} items.')
                logger.info(f'Parse Server fetched {len(memory_items_parse)} items.')
                logger.info(f'Parse Server missing {len(missing_memory_ids)} items.')

                # Normalize Parse Server items
                normalized_memory_items: List[ParseStoredMemory] = self.normalize_and_merge_memory_items(
                    similar_memory_items_neo, 
                    memory_items_parse
                )
                
                # Process final items with chunk handling
                final_memory_items: List[ParseStoredMemory] = []
                for item in normalized_memory_items:
                    if item.memoryId in chunk_id_mapping:
                        # Create a copy of the item to modify
                        updated_item = item.model_copy()
                        updated_item.memoryId = chunk_id_mapping[item.memoryId]
                        
                        # Handle matching chunks if needed
                        if item.memoryChunkIds:
                            matching_chunks = [
                                chunk_id for chunk_id in item.memoryChunkIds 
                                if chunk_id in memory_item_ids
                            ]
                            if matching_chunks:
                                # Add matching chunks if we have them
                                setattr(updated_item, 'matchingChunkIds', matching_chunks)
                        
                        final_memory_items.append(updated_item)
                    else:
                        final_memory_items.append(item)

                logger.info(f'Final Memory Items Count: {len(final_memory_items)}')

                # Validation is handled by ParseStoredMemory model, but we can still log if needed
                items_without_content = [
                    item.memoryId for item in final_memory_items 
                    if not item.content
                ]
                if items_without_content:
                    logger.error(f"Final memory items missing content: {items_without_content}")

                return final_memory_items
            
            except Exception as e:
                logger.error(f"Error in fetch_memory_items_from_sources_async: {e}")
                return []

       

   
    async def get_pinecone_related_memories_async(self, query_embedding, acl_filter: dict, top_k=15):
        """
        Async method to get related memories from Pinecone
        
        Args:
            query_embedding: The embedding vector to query
            acl_filter (dict): Dictionary containing ACL filter conditions
            top_k (int): Number of results to return
            
        Returns:
            Dict containing matches with their metadata and scores
        """
        try:
            # Use regular query since Pinecone doesn't have async methods
            results = self.index.query(
                namespace="",
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                vector=query_embedding,
                filter=acl_filter
            )


            # Convert QueryResponse to a standard dictionary format
            processed_results = {
                'matches': [
                    {
                        'id': match.id,
                        'score': match.score,
                        'metadata': match.metadata,
                        'values': match.values if hasattr(match, 'values') else None
                    }
                    for match in results.matches
                ]
            }
            
            return processed_results
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return {'matches': []}
        
    async def get_bigbird_related_memories_async(self, query_embedding, acl_filter: dict, user_id: str, top_k: int=10):
        """
        Async method to get related memories from BigBird index
        
        Args:
            query_embedding: The embedding vector to query
            acl_filter (dict): Dictionary containing ACL filter conditions
            user_id (str): The user ID
            top_k (int): Number of results to return
            
        Returns:
            List of memory items with their metadata and scores
        """
        try:
            # Use regular query since Pinecone doesn't have async methods
            results = self.bigbird_index.query(
                namespace="",
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                vector=query_embedding,
                filter=acl_filter
            )
            
            # Convert QueryResponse to a standard dictionary format
            bigbird_memory_info = [
                {
                    'id': match.id,
                    'metadata': match.metadata,
                    'score': match.score
                }
                for match in results.matches
            ]
            
            return bigbird_memory_info
        except Exception as e:
            logger.error(f"Error in get_bigbird_related_memories_async: {e}")
            return []

   
    async def get_user_memGraph_schema_neo_async(
        self,
        user_id: str,
        acl_filter: Dict[str, Any] = None,
        neo_session: Optional[AsyncSession] = None,
        timeout_seconds: int = 180
    ) -> Dict[str, List[str]]:
        """
        Get the user's memory graph schema from Neo4j using ACL filter - async version
        
        Args:
            user_id: The user's ID
            acl_filter: Dictionary containing ACL filter conditions
            neo_session: Optional existing Neo4j session to reuse
            timeout_seconds: Maximum time to wait for query execution
        Returns:
            Dict containing lists of nodes and relationships
        """
        # Query to get distinct node labels with ACL filtering
        node_labels_query = """
        MATCH (n)
        WHERE (
            n.user_id = $user_id OR
            any(x IN n.user_read_access WHERE x IN $user_read_access) OR
            any(x IN n.workspace_read_access WHERE x IN $workspace_read_access) OR
            any(x IN n.role_read_access WHERE x IN $role_read_access)
        )
        RETURN DISTINCT labels(n) AS labels
        """
        
        relationship_types_query = """
        MATCH (m)-[r]->(n)
        WHERE (
            m.user_id = $user_id OR
            any(x IN m.user_read_access WHERE x IN $user_read_access) OR
            any(x IN m.workspace_read_access WHERE x IN $workspace_read_access) OR
            any(x IN m.role_read_access WHERE x IN $role_read_access)
        )
        RETURN DISTINCT type(r) AS type
        """

        try:
            # Only ensure connection if we don't have a session
            if not neo_session:
                await self.ensure_async_connection()
                if self.async_neo_conn.fallback_mode:
                    logger.warning("Neo4j in fallback mode, cannot get schema")
                    return {'nodes': [], 'relationships': []}
                
            # Extract filter values from acl_filter
            filter_params = {
                "user_id": user_id,
                "user_read_access": [],
                "workspace_read_access": [],
                "role_read_access": []
            }

            if acl_filter and "$or" in acl_filter:
                for condition in acl_filter["$or"]:
                    if "user_read_access" in condition:
                        filter_params["user_read_access"] = condition["user_read_access"].get("$in", [])
                    elif "workspace_read_access" in condition:
                        filter_params["workspace_read_access"] = condition["workspace_read_access"].get("$in", [])
                    elif "role_read_access" in condition:
                        filter_params["role_read_access"] = condition["role_read_access"].get("$in", [])
            
            valid_nodes: Set[str] = set()
            valid_relationships: List[str] = []

            async with contextlib.AsyncExitStack() as stack:
                # Use existing session or create new one
                session = neo_session or await stack.enter_async_context(self.async_neo_conn.get_session())

                try:
                    # Add timeout context
                    async with asyncio.timeout(timeout_seconds):
                        # Execute queries asynchronously
                        node_labels_result = await session.run(node_labels_query, filter_params)
                        relationship_types_result = await session.run(relationship_types_query, filter_params)

                        # Process node labels
                        async for record in node_labels_result:
                            labels = record.get("labels", [])
                            for label in labels:
                                if label == "Bug":
                                    valid_nodes.add("Task")  # Map Bug to Task
                                else:
                                    try:
                                        # Validate the label
                                        node_label = NodeLabel(label)
                                        valid_nodes.add(label)
                                    except ValueError:
                                        logger.warning(f"Skipping invalid node label: {label}")

                        # Process relationship types
                        async for record in relationship_types_result:
                            rel_type = record.get("type")
                            if rel_type and rel_type in [r.value for r in RelationshipType]:
                                valid_relationships.append(rel_type)

                        # Consume results
                        await node_labels_result.consume()
                        await relationship_types_result.consume()

                except asyncio.TimeoutError:
                    logger.error(f"Neo4j schema query timed out after {timeout_seconds} seconds")
                    self.async_neo_conn.fallback_mode = True
                    self.async_neo_conn.last_fallback_time = time.time()
                    return {'nodes': [], 'relationships': []}
                except Exception as e:
                    logger.error(f"Error executing Neo4j schema queries: {str(e)}")
                    return {'nodes': [], 'relationships': []}

                logger.info(f'Processed nodes: {valid_nodes}')
                logger.info(f'Processed relationships: {valid_relationships}')

                return {
                    'nodes': list(valid_nodes),
                    'relationships': valid_relationships
                }

        except Exception as e:
            logger.error(f"Error in get_user_memGraph_schema_neo_async: {str(e)}")
            logger.exception(e)
            return {'nodes': [], 'relationships': []}
    
    def convert_sets_to_lists(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets_to_lists(i) for i in obj]
        return obj


    
    async def query_neo4j_with_user_query_async(
        self, 
        session_token: str, 
        query_context_combined: str, 
        bigbird_memory_ids: list, 
        acl_filter: Dict[str, Any],  
        user_id: str, 
        chat_gpt: "ChatGPTCompletion", 
        project_id: str = None, 
        top_k: int = 10,
        neo_session: Optional[AsyncSession] = None
    ) -> Tuple[List[MemoryNodeProperties], List[NeoNode], str, str]:
        start_total = time.time()
        try:
            # Connection check timing
            connection_start = time.time()
            if not neo_session:
                await self.ensure_async_connection()
                logger.warning(f"Neo4j connection check took: {time.time() - connection_start:.2f}s")
                if self.async_neo_conn.fallback_mode:
                    logger.warning("Neo4j in fallback mode, falling back to empty results")
                    return [], [], None, ""
            
            # Session creation timing
            session_start = time.time()
            async with contextlib.AsyncExitStack() as stack:
                # Use existing session or create new one
                session = neo_session or await stack.enter_async_context(self.async_neo_conn.get_session())
                logger.warning(f"Session creation took: {time.time() - session_start:.2f}s")

                # Get the user's memory graph schema from Neo4j - convert to async
                # Schema retrieval timing
                schema_start = time.time()
                memory_graph_schema = await self.get_user_memGraph_schema_neo_async(
                    user_id, 
                    acl_filter,
                    neo_session=session  
                )
                logger.info(f'query_neo4j_with_user_query_async memory_graph_schema: {memory_graph_schema}')
                logger.warning(f"Schema retrieval and conversion took: {time.time() - schema_start:.2f}s")

                # Memory ID processing timing
                process_ids_start = time.time()
                memory_graph_converted = self.convert_sets_to_lists(memory_graph_schema)
                logger.info(f'bigbird_memory_ids: {bigbird_memory_ids}')
               

                # Process memory IDs to handle both legacy and chunked formats
                processed_memory_ids : List[str] = []
                for memory_id in bigbird_memory_ids:
                    # Check if the ID has a chunk suffix (_#)
                    if '_' in memory_id:
                        split_result: str = memory_id.rsplit('_', maxsplit=1)
                        base_id: str = split_result[0] if len(split_result) > 0 else memory_id
                        processed_memory_ids.append(base_id)
                    else:
                        processed_memory_ids.append(memory_id)
                
                # Remove duplicates while preserving order
                processed_memory_ids = list(dict.fromkeys(processed_memory_ids))
                logger.debug(f'Processed memory IDs: {processed_memory_ids}')

                logger.warning(f"Memory ID processing took: {time.time() - process_ids_start:.2f}s")

                # Generate the Neo4j cipher query - convert to async if needed
                # LLM Query generation timing
                llm_start = time.time()
                cipher_query, is_llm_generated = await chat_gpt.generate_neo4j_cipher_query_async(
                    user_query=query_context_combined,
                    bigbird_memory_ids=processed_memory_ids,
                    acl_filter=acl_filter,
                    context=None,
                    project_id=project_id,
                    user_id=user_id,
                    memory_graph=memory_graph_converted,
                    top_k=top_k
                )
                logger.warning(f"LLM query generation took: {time.time() - llm_start:.2f}s")

                # ACL and parameter preparation timing
                param_start = time.time()
                # Safely extract ACL parameters with type checking
                acl_or_conditions: List[Dict[str, Any]] = acl_filter.get('$or', [])
                
                # Initialize default empty lists for ACL parameters
                user_read_access: List[str] = []
                workspace_read_access: List[str] = []
                role_read_access: List[str] = []
                
                # Safely extract values from ACL conditions
                for condition in acl_or_conditions:
                    if 'user_read_access' in condition:
                        user_read_access = condition['user_read_access'].get('$in', [])
                    elif 'workspace_read_access' in condition:
                        workspace_read_access = condition['workspace_read_access'].get('$in', [])
                    elif 'role_read_access' in condition:
                        role_read_access = condition['role_read_access'].get('$in', [])

                # Prepare parameters for the query
                cipher_relationship_types = memory_graph_converted.get('relationships', []) if memory_graph_converted else []
                sanitized_relationship_types = [rel.replace('-', '_') for rel in cipher_relationship_types] if cipher_relationship_types else []

                # Extract ACL parameters from the dictionary structure based on query type
                parameters: Dict[str, Any] = {
                    'top_k': top_k,
                    'user_id': user_id,
                    'user_read_access': user_read_access,
                    'workspace_read_access': workspace_read_access,
                    'role_read_access': role_read_access
                }

                # Add additional parameters only for fallback query
                if not is_llm_generated:
                    parameters.update({
                        'bigbird_memory_ids': processed_memory_ids,
                        'sanitized_relationship_types': sanitized_relationship_types
                    })

                logger.info(f'Query type: {"LLM Generated" if is_llm_generated else "Fallback"}')
                logger.info(f'Cipher query: {cipher_query}')
                logger.info(f'Parameters: {parameters}')
                logger.warning(f"Parameter preparation took: {time.time() - param_start:.2f}s")

                # Execute query using async session
                try:
                    query_start_time = time.time()
                    # Add timeout context (180 seconds)
                    async with asyncio.timeout(180):
                        records = await session.run(cipher_query, parameters)
                    
                        memory_nodes: List[MemoryNodeProperties] = []
                        neo_nodes: List[NeoNode] = []
                        paths: List[GraphPath] = []

                        record_count = 0
                        
                        async for record in records:
                            record_count += 1
                            logger.debug(f"\n=== Processing record {record_count} ===")
                            
                            # Log the raw record structure
                            logger.debug(f"Record keys: {record.keys()}")
                            logger.debug(f"Raw record data: {record.data()}")  # Add this to see full record
                            
                            # Check if we have a 'result' key or if the path is directly in the record
                            result_data = record.get('result', record)  # Fallback to record if no 'result' key
                            
                            if 'path' in result_data:
                                path_data = result_data['path']
                                logger.debug(f"Path data type: {type(path_data)}")
                                logger.debug(f"Path data attributes: {dir(path_data)}")
                                
                                # Extract nodes from path
                                if hasattr(path_data, 'nodes'):
                                    for node in path_data.nodes:
                                        node_dict = dict(node.items())
                                        node_labels = list(node.labels)
                                        primary_label = node_labels[0] if node_labels else None
                                        
                                        logger.debug(f"\nProcessing node from path:")
                                        logger.debug(f"Labels: {node_labels}")
                                        logger.debug(f"Properties: {json.dumps(node_dict, indent=2)}")
                                        
                                        try:
                                            converted_node = NodeConverter.convert_to_neo_node(node_dict, primary_label)
                                            if converted_node:
                                                if primary_label == 'Memory':
                                                    memory_nodes.append(converted_node.properties)
                                                else:
                                                    neo_nodes.append(converted_node)
                                                logger.debug(f"✓ Successfully added {primary_label} node")
                                            else:
                                                logger.warning(f"Failed to convert {primary_label} node")
                                                
                                        except ValidationError as e:
                                            logger.error(f"Validation error for {primary_label} node:")
                                            logger.error(f"Error details: {str(e)}")
                                            logger.error(f"Failed properties: {json.dumps(node_dict, indent=2)}")
                                            continue
                                        except Exception as e:
                                            logger.error(f"Unexpected error processing {primary_label} node: {str(e)}")
                                            logger.error(f"Node data: {json.dumps(node_dict, indent=2)}")
                                            continue

                                # Process relationships from path
                                if hasattr(path_data, 'relationships'):
                                    segments = []
                                    for i in range(len(path_data.relationships)):
                                        rel = path_data.relationships[i]
                                        start_node_dict = dict(path_data.nodes[i].items())
                                        end_node_dict = dict(path_data.nodes[i + 1].items())

                                        # Get labels for both nodes
                                        start_node_labels = list(path_data.nodes[i].labels)
                                        end_node_labels = list(path_data.nodes[i + 1].labels)
                                        
                                        start_primary_label = start_node_labels[0] if start_node_labels else None
                                        end_primary_label = end_node_labels[0] if end_node_labels else None
                                        
                                        logger.debug("=== Node Label Debug ===")
                                        logger.debug(f"Start Node Primary Label: {start_primary_label}")
                                        logger.debug(f"Start Node Properties: {json.dumps(start_node_dict, indent=2)}")
                                        logger.debug(f"Relationship Type: {rel.type}")
                                        logger.debug(f"End Node Primary Label: {end_primary_label}")
                                        logger.debug(f"End Node Properties: {json.dumps(end_node_dict, indent=2)}")
                                        
                                        try:
                                            # Convert start node
                                            converted_start_node = NodeConverter.convert_to_neo_node(start_node_dict, start_primary_label)
                                            if not converted_start_node:
                                                logger.error(f"Failed to convert start node with label {start_primary_label}")
                                                continue
                                                
                                            # Convert end node
                                            converted_end_node = NodeConverter.convert_to_neo_node(end_node_dict, end_primary_label)
                                            if not converted_end_node:
                                                logger.error(f"Failed to convert end node with label {end_primary_label}")
                                                continue
                                            
                                            # Create PathSegment with converted nodes
                                            segment = PathSegment(
                                                start_node=converted_start_node.properties,  # Use the properties from NeoNode
                                                relationship=rel.type,
                                                end_node=converted_end_node.properties      # Use the properties from NeoNode
                                            )
                                            segments.append(segment)
                                            logger.info(f"✓ Successfully added path segment: {start_primary_label}-[{rel.type}]-{end_primary_label}")
                                            
                                        except Exception as e:
                                            logger.error(f"Error creating path segment: {str(e)}")
                                            logger.error(f"Start node label: {start_primary_label}, End node label: {end_primary_label}")
                                            continue
                                    # Add this block to create and append the path
                                    if segments:
                                        try:
                                            path = GraphPath(
                                                segments=segments,
                                                length=len(segments)  # Add the required length field
                                            )
                                            paths.append(path)
                                            logger.debug(f"✓ Successfully created path with {len(segments)} segments")
                                        except Exception as e:
                                            logger.error(f"Error creating GraphPath: {str(e)}")
                        
                        query_time = time.time() - query_start_time
                        logger.warning(f"\n=== Query Performance ===")
                        logger.warning(f"Total query execution time: {query_time:.2f} seconds")
                        logger.info(f"Records processed: {record_count}")
                        logger.info(f"Memory nodes found: {len(memory_nodes)}")
                        logger.info(f"Neo nodes found: {len(neo_nodes)}")
                        logger.info(f"Paths created: {len(paths)}")
                        
                        # Result processing timing
                        process_start = time.time()
                        # Create QueryResult for context generation
                        query_result = QueryResult(
                            paths=paths,
                            query=cipher_query
                        )
                    
                        # Get human-readable context
                        text_context = query_result.get_related_context()
                        logger.info(f"Generated context: {text_context}")
                        
                        # Remove duplicates while preserving order
                        memory_nodes = list({node.id: node for node in memory_nodes}.values())
                        neo_nodes = list({node.properties.id: node for node in neo_nodes}.values())
                        
                        logger.info(f"Found {len(memory_nodes)} memory nodes")
                        logger.info(f"Found {len(neo_nodes)} neo nodes")
                        if neo_nodes:
                            logger.info("Sample of neo nodes (first 3):")
                            for i, node in enumerate(neo_nodes[:3]):
                                logger.info(f"Node {i + 1}:")
                                logger.info(f"  Label: {node.label.value}")
                                logger.info(f"  Properties: {json.dumps(node.properties.model_dump(), indent=2)}")
                        logger.info(f"Generated context: {text_context}")

                        logger.warning(f"Result processing took: {time.time() - process_start:.2f}s")

                        total_time = time.time() - start_total
                        logger.warning(f"Total Neo4j query pipeline took: {total_time:.2f}s")
                        
                        return memory_nodes, neo_nodes, cipher_query, text_context
                
                except asyncio.TimeoutError:
                    logger.error("Neo4j query timed out after 180 seconds")
                    self.async_neo_conn.fallback_mode = True
                    self.async_neo_conn.last_fallback_time = time.time()
                    return [], [], None, ""

                except Exception as e:
                    logger.error(f"Error executing Neo4j query: {str(e)}")
                    logger.error(f"Failed query: {cipher_query}")
                    logger.error(f"Failed parameters: {parameters}")
                    return [], [], None, ""

        except Exception as e:
            logger.error(f"Error in query_neo4j_with_user_query_async: {str(e)}")
            return [], [], None, ""
    
    def rank_combined_results(self, combined_results):
        # Implement a ranking algorithm based on relevance to the query
        # This could involve similarity scores from Pinecone and heuristic scores for BigBird groups
        pass
    
    async def find_related_memories(
        self, 
        session_token: str, 
        memory_graph, 
        memory_item: Dict[str, Any],
        queries: List[str], 
        user_id: str, 
        chat_gpt: "ChatGPTCompletion", 
        metadata, 
        skip_neo: bool = True, 
        exclude_memory_id: str = None,
        user_workspace_ids: Optional[List[str]] = None
    ) -> List[ParseStoredMemory]:

        input_memory_id = memory_item.get('memoryId') or memory_item.get('id') or memory_item.get('objectId')
        if not input_memory_id:
            logger.warning(f"Could not find memory ID in input memory item: {memory_item}")

        # Execute all queries in parallel and extract memory_items from results
        query_results = await asyncio.gather(*[
            self.find_related_memory_items_async(
                session_token=session_token,
                query=query,
                context=[],
                user_id=user_id,
                chat_gpt=chat_gpt,
                metadata=metadata,
                relation_type=None,
                project_id=None,
                skip_neo=skip_neo,
                exclude_memory_id=exclude_memory_id,
                user_workspace_ids=user_workspace_ids
            )
            for query in queries
        ])
        
        # Extract just the memory_items from each RelatedMemoryResult
        all_related_memories: List[ParseStoredMemory] = []
        for idx, result in enumerate(query_results):
            memory_items = result.memory_items
            logger.info(f'Query {idx + 1} returned {len(memory_items)} items')
            
            # Filter and get top item
            filtered_memories = [
                item for item in memory_items 
                if input_memory_id and item.memoryId != input_memory_id
            ]
            
            if filtered_memories:
                all_related_memories.append(filtered_memories[0])

        # Remove duplicates
        seen_ids: Set[str] = set()
        seen_contents: Set[str] = set()
        unique_memories: List[ParseStoredMemory] = []
        
        for memory in all_related_memories:
            if not (memory.memoryId in seen_ids or memory.content in seen_contents):
                seen_ids.add(memory.memoryId)
                seen_contents.add(memory.content)
                unique_memories.append(memory)

        logger.info(f'Found {len(all_related_memories)} total memories, {len(unique_memories)} unique memories')
        return unique_memories

    async def delete_memory_item(
        self, 
        memory_id: str, 
        session_token: str,
        skip_parse: bool = False
    ) -> Union[DeleteMemoryResponse, ErrorDetail]:
        """
        Asynchronously deletes a memory item from storage systems.
        
        Args:
            memory_id (str): The ID of the memory item to delete
            session_token (str): Session token for Parse Server authentication
            skip_parse (bool): If True, skip Parse Server deletion (default: False)
        
        Returns:
            Union[DeleteMemoryResponse, ErrorDetail]: DeleteMemoryResponse for success or ErrorDetail for errors
        """
        deletion_status = DeletionStatus()
        exists_somewhere = False
        parse_object_id = ''
        
        try:
            if not memory_id:
                return ErrorDetail(
                    code=400,
                    detail='Memory ID is required'
                )
            
            # Get memory information based on skip_parse flag
            memory_chunk_ids = []
            if skip_parse:
                # Get memory info from Neo4j
                neo4j_memory = await self.get_memory_item(memory_id)
                if neo4j_memory:
                    exists_somewhere = True
                    memory_chunk_ids = neo4j_memory.get('memoryChunkIds', [memory_id])
                    deletion_status.parse = True
            else:
                # Get from Parse Server as before
                parse_memory_item = await retrieve_memory_item_by_pinecone_id(
                    session_token, 
                    str(memory_id)
                )
                if parse_memory_item:
                    exists_somewhere = True
                    parse_object_id = parse_memory_item.get('objectId')
                    memory_chunk_ids = parse_memory_item.get('memoryChunkIds', [memory_id])

            # Ensure we have at least one ID to delete by using memory_id as fallback
            if not memory_chunk_ids:
                logger.warning(f"No memory chunk IDs found for {memory_id}, using memory_id as fallback")
                memory_chunk_ids = [memory_id, f"{memory_id}_0"]
            else:
                # Check if we have only one ID without chunk suffix
                if len(memory_chunk_ids) == 1 and not any(id.endswith(f"_{i}") for id in memory_chunk_ids for i in range(10)):
                    base_id = memory_chunk_ids[0]
                    logger.info(f"Single memory ID without chunk suffix found: {base_id}, adding chunked version")
                    memory_chunk_ids.append(f"{base_id}_0")
                
                # Ensure memory_id is included in chunk_ids if not already present
                if memory_id not in memory_chunk_ids:
                    memory_chunk_ids.append(memory_id)

            logger.info(f"Using memory chunk IDs for deletion: {memory_chunk_ids}")

            # Check Pinecone existence using memory chunk IDs
            try:
                fetch_response = self.index.fetch(ids=memory_chunk_ids)
                if any(chunk_id in fetch_response.vectors for chunk_id in memory_chunk_ids):
                    exists_somewhere = True
            except Exception as e:
                logger.error(f"Error checking Pinecone: {e}")
                
            # Return 404 if memory doesn't exist anywhere
            if not exists_somewhere:
                return ErrorDetail(
                    code=404,
                    detail=f'Memory item with ID {memory_id} not found in any system'
                )

            # Delete from Pinecone
            try:
                pinecone_result = await asyncio.to_thread(
                    self.index.delete,
                    ids=memory_chunk_ids
                )
                logger.info(f'Deleted from Pinecone: {pinecone_result}')
                deletion_status.pinecone = True
            except Exception as e:
                logger.error(f"Error deleting from Pinecone: {e}")
                logger.error("Full traceback:", exc_info=True)

            # Delete from Neo4j with proper async connection handling
            try:
                # Ensure async connection is established
                await self.ensure_async_connection()
                if self.async_neo_conn.fallback_mode:
                    logger.warning("Neo4j in fallback mode, skipping Neo4j deletion")
                    deletion_status.neo4j = True
                    return
                
                async with self.async_neo_conn.get_session() as session:
                    # Execute the delete operation
                    try:
                        # First verify the node exists
                        verify_result = await session.run(
                            "MATCH (m:Memory {id: $id}) RETURN m",
                            {"id": memory_id}
                        )
                        verify_records = await verify_result.values()
                        
                        if verify_records:
                            # Node exists, proceed with deletion
                            delete_result = await session.run(
                                "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                                {"id": memory_id}
                            )
                            await delete_result.consume()  # Ensure the operation completes
                            logger.info('Deleted from Neo4j')
                            deletion_status.neo4j = True
                        else:
                            logger.warning(f"Memory node with ID {memory_id} not found in Neo4j")
                            deletion_status.neo4j = True
                    except Exception as e:
                        logger.error(f"Error in Neo4j operation: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error deleting from Neo4j: {e}", exc_info=True)

            # Delete from Parse Server
            try:
                # Delete from Parse Server only if not skipped
                if not skip_parse:
                    try:
                        parse_result = await delete_memory_item_parse(parse_object_id)
                        if parse_result:
                            logger.info(f'Successfully deleted from Parse Server: {memory_id}')
                            deletion_status.parse = True
                        else:
                            logger.error(f'Failed to delete from Parse Server: {memory_id}')
                    except Exception as e:
                        logger.error(f"Error deleting from Parse Server: {e}")
                else:
                    # Mark Parse deletion as successful when skipped
                    deletion_status.parse = True
                    logger.info(f'Skipped Parse Server deletion for: {memory_id}')

            except Exception as e:
                logger.error(f"Error deleting from Parse Server: {e}")

            # Check if all deletions were successful
            if all(vars(deletion_status).values()):
                return DeleteMemoryResponse(
                    message='Memory item successfully deleted from all systems',
                    memoryId=memory_id,
                    objectId=parse_object_id,
                    status=deletion_status,
                    code='SUCCESS',
                    status_code=200
                )
            else:
                return DeleteMemoryResponse(
                    error='Memory item deletion partially successful',
                    memoryId=memory_id,
                    objectId=parse_object_id,
                    status=deletion_status,
                    code='PARTIAL_DELETE',
                    status_code=207
                )

        except Exception as e:
            logger.error(f"Unexpected error in delete_memory_item: {e}", exc_info=True)
            return ErrorDetail(
                code=500,
                detail=f'Unexpected error: {str(e)}'
            )

    async def get_memory_item(
        self, 
        memory_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Asynchronously fetch a memory item from Neo4j by its ID.

        Args:
            memory_id (str): The ID of the memory item to retrieve

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing memory item properties if found,
                                    None if not found or error occurs
        """
        try:
            # Ensure Neo4j connection is initialized
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, cannot get memory item")
                return None
            #driver = await self.async_neo_conn.get_driver()

            #async with driver.session() as session:
            async with self.async_neo_conn.get_session() as session:
                # Fetch the memory item from the Neo4j graph
                cypher_query = """
                    MATCH (m:Memory)
                    WHERE m.id = $memory_id
                    RETURN m
                """
                parameters = {'memory_id': str(memory_id)}
                
                result = await session.run(cypher_query, parameters)
                record = await result.single()

                if record is None:
                    logger.info(f"No memory item found with ID: {memory_id}")
                    return None

                # Extract and return the properties of the memory item
                memory_properties = dict(record[0])
                logger.info(f"Memory item properties: {memory_properties}")

                return memory_properties

        except Exception as e:
            logger.error(f"Error fetching memory item from Neo4j: {e}")
            logger.error("Full traceback:", exc_info=True)
            return None
                
    async def update_memory_item(
        self,
        session_token: str,
        memory_id: str,
        memory_type: str,
        content: str,
        metadata: dict,
        background_tasks: BackgroundTasks,
        context: str = None
    ) -> UpdateMemoryResponse:
        """
        Update a memory item in all storage systems (Pinecone, Neo4j, Parse Server).
        
        Args:
            session_token (str): The session token for authentication
            memory_id (str): The ID of the memory item to update
            memory_type (str): The type of memory item
            content (str): The new content for the memory item
            metadata (dict): The new metadata for the memory item
            background_tasks (BackgroundTasks): FastAPI background tasks object
            context (str, optional): The context for the memory item
            
        Returns:
            UpdateMemoryResponse: The response containing update status and updated item
        """
        status = SystemUpdateStatus()
        
        try:
            # Get the existing memory item
            existing_item = await self.get_memory_item(memory_id)
            if not existing_item:
                return UpdateMemoryResponse.error_response(
                    "Memory item not found",
                    code=404
                )

            # Extract the properties of the memory item
            # The existing_item is already a dictionary from Neo4j
            memory_properties = existing_item
            is_metadata_only_update = content is None or content == ""

            existing_memory_type = memory_properties.get('type')
            logger.info(f"existing_memory_type: {existing_memory_type}")
            logger.info(f"existing_item fron neo: {existing_item}")
            memory_type = memory_type if memory_type else existing_memory_type
     
            memory_properties['metadata'] = memory_properties.get('metadata', {})
            if context is not None:
                memory_properties['context'] = context
            
            if isinstance(memory_properties['metadata'], str):
                try:
                    memory_properties['metadata'] = json.loads(memory_properties['metadata'])
                except json.JSONDecodeError:
                    memory_properties['metadata'] = {}
            memory_properties['metadata'].update(metadata)

            # Sanitize metadata
            memory_properties['metadata'] = {
                k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
                for k, v in memory_properties['metadata'].items()
            }

            logger.info(f"Updated memory properties: {memory_properties}")

            # Get the vector ID from memoryChunkIds
            memory_chunk_ids = memory_properties.get('memoryChunkIds', [])
            if isinstance(memory_chunk_ids, str):
                try:
                    memory_chunk_ids = json.loads(memory_chunk_ids)
                except json.JSONDecodeError:
                    memory_chunk_ids = [id.strip() for id in memory_chunk_ids.split(',') if id.strip()]
            elif isinstance(memory_chunk_ids, list):
                memory_chunk_ids = [str(id).strip().strip("'[]\"") for id in memory_chunk_ids]
            else:
                memory_chunk_ids = []

            if not memory_chunk_ids:
                logger.warning(f"No memoryChunkIds found for memory {memory_id}, using memory_id as fallback")
                vector_id = str(memory_id)
            else:
                # Get the first chunk ID since we're dealing with a single chunk
                vector_id = str(memory_chunk_ids[0])

            if is_metadata_only_update:
                # For metadata-only updates (like ACL changes)
                logger.info("Performing metadata-only update")
                
                # Update the memory properties with new ACL values from metadata by appending rather than overwriting
                if 'user_read_access' in memory_properties['metadata']:
                    existing_user_read = set(memory_properties.get('user_read_access', []))
                    logger.info(f"existing_user_read: {existing_user_read}")
                    new_user_read = set(memory_properties['metadata']['user_read_access'])
                    memory_properties['user_read_access'] = list(existing_user_read | new_user_read)
                    # Update metadata to match
                    memory_properties['metadata']['user_read_access'] = memory_properties['user_read_access']
                    logger.info(f"memory_properties['user_read_access']: {memory_properties['user_read_access']}")

                if 'user_write_access' in memory_properties['metadata']:
                    existing_user_write = set(memory_properties.get('user_write_access', []))
                    logger.info(f"existing_user_write: {existing_user_write}")
                    new_user_write = set(memory_properties['metadata']['user_write_access'])
                    memory_properties['user_write_access'] = list(existing_user_write | new_user_write)
                    # Update metadata to match
                    memory_properties['metadata']['user_write_access'] = memory_properties['user_write_access']
                    logger.info(f"memory_properties['user_write_access']: {memory_properties['user_write_access']}")

                if 'workspace_read_access' in memory_properties['metadata']:
                    existing_workspace_read = set(memory_properties.get('workspace_read_access', []))
                    new_workspace_read = set(memory_properties['metadata']['workspace_read_access'])
                    memory_properties['workspace_read_access'] = list(existing_workspace_read | new_workspace_read)
                    # Update metadata to match
                    memory_properties['metadata']['workspace_read_access'] = memory_properties['workspace_read_access']
                    logger.info(f"memory_properties['workspace_read_access']: {memory_properties['workspace_read_access']}")

                if 'workspace_write_access' in memory_properties['metadata']:
                    existing_workspace_write = set(memory_properties.get('workspace_write_access', []))
                    new_workspace_write = set(memory_properties['metadata']['workspace_write_access'])
                    memory_properties['workspace_write_access'] = list(existing_workspace_write | new_workspace_write)
                    # Update metadata to match
                    memory_properties['metadata']['workspace_write_access'] = memory_properties['workspace_write_access']
                    logger.info(f"memory_properties['workspace_write_access']: {memory_properties['workspace_write_access']}")

                if 'role_read_access' in memory_properties['metadata']:
                    existing_role_read = set(memory_properties.get('role_read_access', []))
                    new_role_read = set(memory_properties['metadata']['role_read_access'])
                    memory_properties['role_read_access'] = list(existing_role_read | new_role_read)
                    # Update metadata to match
                    memory_properties['metadata']['role_read_access'] = memory_properties['role_read_access']
                    logger.info(f"memory_properties['role_read_access']: {memory_properties['role_read_access']}")

                if 'role_write_access' in memory_properties['metadata']:
                    existing_role_write = set(memory_properties.get('role_write_access', []))
                    new_role_write = set(memory_properties['metadata']['role_write_access'])
                    memory_properties['role_write_access'] = list(existing_role_write | new_role_write)
                    # Update metadata to match
                    memory_properties['metadata']['role_write_access'] = memory_properties['role_write_access']
                    logger.info(f"memory_properties['role_write_access']: {memory_properties['role_write_access']}")


                logger.info(f"memory_properties: {memory_properties}")
                
                # Get all chunk IDs for this memory
                memory_chunk_ids = memory_properties.get('memoryChunkIds', [])
                if isinstance(memory_chunk_ids, str):
                    try:
                        memory_chunk_ids = json.loads(memory_chunk_ids)
                    except json.JSONDecodeError:
                        memory_chunk_ids = [id.strip() for id in memory_chunk_ids.split(',') if id.strip()]
                elif isinstance(memory_chunk_ids, list):
                    memory_chunk_ids = [str(id).strip().strip("'[]\"") for id in memory_chunk_ids]
                
                # If no chunk IDs found, use memory_id as fallback
                if not memory_chunk_ids:
                    memory_chunk_ids = [str(memory_id)]
                
                # Update metadata in Pinecone for all chunks
                pinecone_success = True
                for chunk_id in memory_chunk_ids:
                    chunk_success = await self.update_pinecone_metadata(
                        vector_id=chunk_id,
                        new_metadata=memory_properties['metadata']
                    )
                    pinecone_success = pinecone_success and chunk_success
                status.pinecone = pinecone_success

                # Update in BigBird (using base memory_id without chunks)
                # don't update bigbird metadata for now, since it might have other memories that this user doesn't have access to
                #try:
                #    await self.update_bigbird_metadata(
                #        memory_id=memory_id,
                #        metadata=memory_properties['metadata']
                #    )
                #except Exception as e:
                #    logger.error(f"Error updating BigBird metadata: {e}")
                
                # Update in Neo4j
                neo4j_result = await self.update_memory_item_in_neo4j(
                    memory_properties, 
                    memory_type
                )
                status.neo4j = neo4j_result is not None

                # Update in Parse
                parse_memory = await retrieve_memory_item_by_pinecone_id(
                    session_token, 
                    vector_id
                )
                if parse_memory:
                    # Convert ACL for Parse update
                    parse_acl = convert_acl(memory_properties['metadata'])

                    logger.info(f"memory_chunk_ids: {memory_chunk_ids}")
                    logger.info(f"parse_memory: {parse_memory}")

                    parse_update_data = {
                        'objectId': parse_memory.get('objectId'),
                        'memoryId': memory_id,
                        'content': parse_memory.get('content', ''),  # Use existing content for metadata-only updates
                        'metadata': memory_properties['metadata'],
                        'memoryChunkIds': memory_chunk_ids,   
                        'ACL': parse_acl  
                    }
                    logger.info(f"parse_update_data: {parse_update_data}")
                    parse_result = await update_memory_item(session_token, parse_update_data)
                    logger.info(f"parse_result: {parse_result}")
                    status.parse = True if parse_result and parse_result.memory_items and len(parse_result.memory_items) > 0 else False
                    
                    if status.parse and parse_result.memory_items:
                        # Extract and verify objectId
                        object_id = parse_result.memory_items[0].objectId
                        logger.info(f"Retrieved objectId from parse_result: {object_id}")

                        updatedAt = parse_result.memory_items[0].updatedAt
                        logger.info(f"Retrieved updatedAt from parse_result: {updatedAt}")

                        updated_item = UpdateMemoryItem(
                            objectId=object_id,
                            memoryId=memory_id,
                            content=parse_memory.get('content', ''),   
                            updatedAt=updatedAt,
                            memoryChunkIds=memory_chunk_ids  # Add this line to include the chunk IDs
                        )
                        
                        logger.info(f"Created UpdateMemoryItem with objectId: {updated_item.objectId} and memoryChunkIds: {memory_chunk_ids}")

                        return UpdateMemoryResponse.success_response(
                            items=[updated_item],
                            status=status
                        )

                # If Parse update failed
                return UpdateMemoryResponse.error_response(
                    "Failed to update in Parse Server",
                    code=500,
                    status=status
                )
            else:
                # Update the content, metadata and context
                memory_properties['content'] = content

                # Generate new embeddings asynchronously
                embeddings, chunks = await self.embedding_model.get_sentence_embedding(content)
                num_embeddings = len(embeddings)

                if num_embeddings == 1:
                    # Ensure embedding is a list of floats
                    embedding = [float(x) for x in embeddings[0]]
                    
                    # Update in Pinecone
                    
                    pinecone_success = await self.update_pinecone(
                        vector_id=vector_id,
                        embedding=embedding,
                        new_metadata=memory_properties['metadata']
                    )
                    status.pinecone = pinecone_success

                    # Update in Neo4j
                    neo4j_result = await self.update_memory_item_in_neo4j(
                        memory_properties, 
                        memory_type
                    )
                    status.neo4j = neo4j_result is not None

                    # Update in Parse
                    parse_memory = await retrieve_memory_item_by_pinecone_id(
                        session_token, 
                        vector_id
                    )
                    logger.info(f"parse_memory from retrieve_memory_item_by_pinecone_id: {parse_memory}")

                    if parse_memory:
                        # Get the objectId directly from the response
                        memory_objectId = parse_memory['objectId'] 
                        logger.info(f"memory_objectId: {memory_objectId}")

                        # Convert ACL for Parse update
                        parse_acl = convert_acl(memory_properties['metadata'])

                        parse_update_data = {
                            'objectId': memory_objectId,
                            'memoryId': memory_id,
                            'content': content,
                            'metadata': memory_properties['metadata'],
                            'memoryChunkIds': memory_chunk_ids,   
                            'ACL': parse_acl  
                        }
                        parse_result = await update_memory_item(session_token, parse_update_data)
                        logger.info(f"parse_result: {parse_result}")
                        status.parse = True if parse_result and parse_result.memory_items and len(parse_result.memory_items) > 0 else False
                        
                        if status.parse and parse_result.memory_items:
                            # Extract and verify objectId
                            object_id = parse_result.memory_items[0].objectId
                            logger.info(f"Retrieved objectId from parse_result: {object_id}")

                            updatedAt = parse_result.memory_items[0].updatedAt
                            logger.info(f"Retrieved updatedAt from parse_result: {updatedAt}")

                            updated_item = UpdateMemoryItem(
                                objectId=object_id,
                                memoryId=memory_id,
                                content=content,
                                updatedAt=updatedAt,
                                memoryChunkIds=memory_chunk_ids  # Add this line to include the chunk IDs
                            )
                            
                            logger.info(f"Created UpdateMemoryItem with objectId: {updated_item.objectId} and memoryChunkIds: {memory_chunk_ids}")

                            return UpdateMemoryResponse.success_response(
                                items=[updated_item],
                                status=status
                            )

                    # If Parse update failed
                    return UpdateMemoryResponse.error_response(
                        "Failed to update in Parse Server",
                        code=500,
                        status=status
                    )

                else:
                    # Handle multiple chunks in background task
                    background_tasks.add_task(
                        self.process_memory_chunks_async,
                        session_token=session_token,
                        memory_id=memory_id,
                        memory_properties=memory_properties,
                        memory_type=memory_type,
                        embeddings=embeddings,
                        chunks=chunks
                    )
                    
                    return UpdateMemoryResponse.success_response(
                        items=[],  # Empty list as processing is happening in background
                        status=SystemUpdateStatus(
                            pinecone=True,
                            neo4j=True,
                            parse=True
                        )
                    )

        except Exception as e:
            logger.error(f"Error in update_memory_item: {e}")
            logger.error("Full traceback:", exc_info=True)
            return UpdateMemoryResponse.error_response(
                str(e),
                code=500
            )

    async def process_memory_chunks_async(
        self,
        session_token: str,
        memory_id: str,
        memory_properties: dict,
        memory_type: str,
        embeddings: List[List[float]],
        chunks: List[str]
    ) -> UpdateMemoryResponse:
        """
        Process multiple memory chunks asynchronously in the background.
        Only creates new chunk IDs in Pinecone if necessary, and updates Neo4j and Parse
        with the consolidated memory item.
        """
        try:
            status = SystemUpdateStatus()
            new_chunk_ids = []

            # First, update the main memory item in Pinecone with first chunk
            pinecone_success = await self.update_pinecone(
                str(memory_id),
                embeddings[0],
                memory_properties['metadata']
            )
            status.pinecone = pinecone_success
            logger.info(f"Updated main memory in Pinecone: {memory_id}")

            # Process additional chunks in Pinecone if any
            if len(chunks) > 1:
                for index, (embedding, chunk) in enumerate(zip(embeddings[1:], chunks[1:]), 1):
                    # Generate new chunk ID
                    chunk_id = f"{memory_id}_chunk_{index}"
                    new_chunk_ids.append(chunk_id)
                    
                    # Add chunk to Pinecone
                    chunk_success = await self.update_pinecone(
                        chunk_id,
                        embedding,
                        memory_properties['metadata']
                    )
                    status.pinecone = status.pinecone and chunk_success
                    logger.info(f"Added chunk to Pinecone: {chunk_id}")

            # Update memoryChunkIds in properties if we created new chunks
            if new_chunk_ids:
                existing_chunk_ids = memory_properties.get('memoryChunkIds', [])
                # Clean existing chunk IDs
                if isinstance(existing_chunk_ids, str):
                    try:
                        existing_chunk_ids = json.loads(existing_chunk_ids)
                    except json.JSONDecodeError:
                        existing_chunk_ids = [id.strip() for id in existing_chunk_ids.split(',') if id.strip()]
                elif isinstance(existing_chunk_ids, list):
                    existing_chunk_ids = [str(id).strip().strip("'[]\"") for id in existing_chunk_ids]
                else:
                    existing_chunk_ids = []
                memory_properties['memoryChunkIds'] = existing_chunk_ids + new_chunk_ids

            # Clean memory chunk IDs one final time before creating parse_update_data
            memory_chunk_ids = memory_properties.get('memoryChunkIds', [])
            if isinstance(memory_chunk_ids, str):
                try:
                    memory_chunk_ids = json.loads(memory_chunk_ids)
                except json.JSONDecodeError:
                    memory_chunk_ids = [id.strip() for id in memory_chunk_ids.split(',') if id.strip()]
            elif isinstance(memory_chunk_ids, list):
                memory_chunk_ids = [str(id).strip().strip("'[]\"") for id in memory_chunk_ids]
            else:
                memory_chunk_ids = []

            # Get parse_memory before trying to use it
            parse_memory = await retrieve_memory_item_by_pinecone_id(
                session_token,
                str(memory_id)
            )
            if not parse_memory:
                return UpdateMemoryResponse.error_response(
                    "Failed to retrieve memory item from Parse Server",
                    code=404,
                    status=status
                )

            parse_update_data = {
                'objectId': parse_memory.get('objectId'),
                'memoryId': memory_id,
                'content': content,
                'metadata': sanitized_metadata,
                'memoryChunkIds': memory_chunk_ids,  # Now using cleaned memory_chunk_ids
                'type': memory_type
            }

            logger.info(f"parse_update_data inside process_memory_chunks_async: {parse_update_data}")

            # Create the appropriate MemoryItem instance based on the type
            sanitized_metadata = {
                k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
                for k, v in memory_properties.get('metadata', {}).items()
            }
            memory_context = memory_properties.get('context', {})
            content = memory_properties.get('content', '')

            memory_item = None
            if memory_type == 'TextMemoryItem':
                memory_item = TextMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'CodeSnippetMemoryItem':
                memory_item = CodeSnippetMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'DocumentMemoryItem':
                memory_item = DocumentMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'WebpageMemoryItem':
                memory_item = WebpageMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'CodeFileMemoryItem':
                memory_item = CodeFileMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'MeetingMemoryItem':
                memory_item = MeetingMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'PluginMemoryItem':
                memory_item = PluginMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'IssueMemoryItem':
                memory_item = IssueMemoryItem(content, sanitized_metadata, memory_context)
            elif memory_type == 'CustomerMemoryItem':
                memory_item = CustomerMemoryItem(content, sanitized_metadata, memory_context)

            if not memory_item:
                return UpdateMemoryResponse.error_response(
                    f"Unsupported memory type: {memory_type}",
                    code=400,
                    status=status
                )

            # Update in Neo4j with consolidated memory item
            neo4j_result = await self.update_memory_item_in_neo4j(
                memory_item.__dict__,
                memory_type
            )
            status.neo4j = neo4j_result is not None
            logger.info(f"Updated memory in Neo4j: {memory_id}")

            # Get parse_memory for Parse Server update
            parse_memory = await retrieve_memory_item_by_pinecone_id(
                session_token,
                str(memory_id)
            )
            if parse_memory:
                # Create chunk IDs for each chunk
                chunk_ids = [f"{memory_id}_{i}" for i in range(len(chunks))]
                
                parse_update_data = {
                    'objectId': parse_memory.get('objectId'),
                    'memoryId': memory_id,
                    'content': content,
                    'metadata': sanitized_metadata,
                    'memoryChunkIds': chunk_ids,  # Add the chunk IDs explicitly
                    'type': memory_type
                }
                logger.info(f"parse_update_data inside process_memory_chunks_async: {parse_update_data}")
                parse_result = await update_memory_item(session_token, parse_update_data)
                status.parse = parse_result is not None
                
                if parse_result:
                    updated_item = UpdateMemoryItem(
                        objectId=parse_result.objectId,
                        memoryId=parse_result.memoryId,
                        content=parse_result.content,
                        updatedAt=parse_result.updatedAt,
                        memoryChunkIds=chunk_ids
                    )
                    return UpdateMemoryResponse.success_response(
                        items=[updated_item],
                        status=status
                    )
                
            return UpdateMemoryResponse.error_response(
                "Failed to update memory item in Parse Server",
                code=500,
                status=status
            )

        except Exception as e:
            logger.error(f"Error processing memory chunks: {e}")
            logger.error("Full traceback:", exc_info=True)
            return UpdateMemoryResponse.error_response(
                str(e),
                code=500,
                status=SystemUpdateStatus()
            )
            
    def find_and_delete_duplicates(self, user_id, session_token):
        # Modified method to find and delete duplicates, then return counts or relevant info
        duplicates = self.identify_duplicates(user_id)
        num_duplicates_found = len(duplicates)
        self.delete_duplicate_memories(duplicates, session_token)
        num_duplicates_deleted = len(duplicates)  # Assuming all found duplicates are deleted successfully
        return num_duplicates_found, num_duplicates_deleted

    def identify_duplicates(self, user_id):
        # Get user info for ACL filter
        user_instance = User.get(user_id)
        user_roles = user_instance.get_roles() if user_instance else []
        user_workspace_ids = User.get_workspaces_for_user(user_id)
        
        # Setup the ACL filter using the working structure
        acl_filter = {
            "$or": [
                {"user_id": {"$eq": str(user_id)}},
                {"user_read_access": {"$in": [str(user_id)]}},
                {"workspace_read_access": {"$in": [str(workspace_id) for workspace_id in user_workspace_ids]}},
                {"role_read_access": {"$in": user_roles}},
            ]
        }
        
        # Fetch all memories' embeddings and IDs for the specified user using ACL filter
        response = self.index.query(
            vector=[0] * 768,  # Dummy vector to get all memories
            filter=acl_filter,
            top_k=1000,  # Adjust based on your needs
            include_values=True
        )
        
        duplicates = []
        checked_ids = set()  # Keep track of memory IDs that have been checked

        # Iterate through each memory to find its nearest neighbors (excluding itself)
        for match in response.matches:
            memory_id = match.id
            embedding = match.values

            if memory_id in checked_ids:
                continue  # Skip if already checked

            # Find similar memories using Pinecone's similarity search with same ACL filter
            similar_memories = self.index.query(
                vector=embedding,
                filter=acl_filter,
                include_metadata=True,
                top_k=20
            )
            
            # Filter out duplicates based on cosine similarity threshold (e.g., > 0.95)
            for sim_memory in similar_memories.matches:
                if sim_memory.score > 0.95 and sim_memory.id != memory_id:
                    duplicates.append(sim_memory.id)

            checked_ids.add(memory_id)

        return list(set(duplicates))  # Return a list of unique duplicate IDs
        return list(set(duplicates))  # Return a list of unique duplicate IDs

    def delete_duplicate_memories(self, duplicate_ids, session_token):
        for memory_id in duplicate_ids:
            # Use Pinecone's delete method to remove the memory
            self.delete_memory_item(memory_id, session_token)
            logger.info(f"Deleted duplicate memory with ID from both pine-cone and neo {memory_id}")

    
   
    async def _node_exists(self, node_id: str, node_type: Optional[NodeLabel] = None, node_content: Optional[str] = None) -> Union[str, bool]:
        """Check if a node exists in Neo4j by ID or content"""
        try:
            # Ensure connection is initialized
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, cannot check if node exists")
                return False
            
            # Get the driver
            driver = await self.async_neo_conn.get_driver()
            
            # Create the appropriate query based on node type and content
            if node_type and node_content:
                # Simplified queries that check by content/name first
                if node_type == NodeLabel.Memory:
                    query = """
                    MATCH (n:Memory) 
                    WHERE n.content = $node_content
                    RETURN n.id as existing_id, COUNT(n) as count
                    """
                elif node_type in [NodeLabel.Person, NodeLabel.Company, NodeLabel.Project]:
                    query = f"""
                    MATCH (n:{node_type.value}) 
                    WHERE n.name = $node_content
                    RETURN n.id as existing_id, COUNT(n) as count
                    """
                else:
                    # For other types, check by ID
                    query = "MATCH (n) WHERE n.id = $node_id RETURN n.id as existing_id, COUNT(n) as count"
            else:
                # If no type/content provided, just check by ID
                query = "MATCH (n) WHERE n.id = $node_id RETURN n.id as existing_id, COUNT(n) as count"

            params = {"node_id": node_id, "node_content": node_content} if node_content else {"node_id": node_id}
            
            logger.debug(f"Node exists query: {query}")
            logger.debug(f"Node exists params: {params}")
            
            #async with driver.session() as session:
            async with self.async_neo_conn.get_session() as session:
                result = await session.run(query, params)
                records = await result.values()
                await result.consume()

                for record in records:
                    if record and len(record) >= 2 and record[1] > 0:
                        logger.info(f"Found existing {node_type.value if node_type else 'node'} with content '{node_content}', id: {record[0]}")
                        return record[0]  # Return existing_id
                return False

        except Exception as e:
            logger.error(f"Error checking if node exists: {e}")
            return False

    async def store_llm_generated_graph(
        self,
        nodes: List[Node],
        relationships: List[Relationship],
        memory_item: Dict[str, Any],
        workspace_id: Optional[str] = None
    ):
        """
        Stores the LLM-generated graph structure in Neo4j, applying proper metadata and access controls.
        """
        # Ensure async connection is initialized
        await self.ensure_async_connection()
        if self.async_neo_conn.fallback_mode:
            logger.warning("Neo4j in fallback mode, cannot store graph")
            return
        
        # Get driver using await
        #driver = await self.async_neo_conn.get_driver()
        
        #async with driver.session() as session:
        async with self.async_neo_conn.get_session() as session:
            try:
                # Get the current event loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                # Make sure async driver is connected
                await self.ensure_async_connection()
                if self.async_neo_conn.fallback_mode:
                    logger.warning("Neo4j in fallback mode, cannot store graph")
                    return
                #driver = await self.async_neo_conn.get_driver()
                
                #async with driver.session() as session:
                async with self.async_neo_conn.get_session() as session:
                    # Extract metadata from memory item
                    # Extract metadata from memory item
                    metadata = json.loads(memory_item['metadata']) if isinstance(memory_item.get('metadata'), str) else memory_item.get('metadata', {})
                    
                    # Common metadata fields to include for all nodes
                    common_metadata = {
                        "user_id": metadata.get("user_id"),
                        "pageId": metadata.get("pageId"),
                        "conversationId": metadata.get("conversationId"),
                        "sourceType": metadata.get("sourceType"),
                        "sourceUrl": metadata.get("sourceUrl"),
                        "workspace_id": workspace_id or metadata.get("workspace_id"),
                        "user_read_access": metadata.get("user_read_access", []),
                        "user_write_access": metadata.get("user_write_access", []),
                        "workspace_read_access": metadata.get("workspace_read_access", []),
                        "workspace_write_access": metadata.get("workspace_write_access", []),
                        "role_read_access": metadata.get("role_read_access", []),
                        "role_write_access": metadata.get("role_write_access", []),
                        "createdAt": metadata.get("createdAt") or datetime.utcnow().isoformat()
                    }

                    # Convert dictionary nodes to Node objects if needed
                    nodes_objects = [
                        node if isinstance(node, Node) else Node(**node)
                        for node in nodes
                    ]

                    # Convert dictionary relationships to Relationship objects if needed  
                    relationship_objects = [
                        rel if isinstance(rel, Relationship) else Relationship(**rel)
                        for rel in relationships
                    ]

                    

                    # Create nodes first
                    for node in nodes_objects:
                        node_id = node.properties.get('id', str(uuid4()))
                        node_content = node.properties.get('content') or node.properties.get('name')

                        logger.info(f"Checking existence of {node.label} node with content: {node_content}")
                    
                        try:
                            # Check if any node type already exists
                            existing_id = await self._node_exists(
                                node_id=node_id,
                                node_type=NodeLabel[node.label],  # Convert string label to NodeLabel enum
                                node_content=node_content
                            )
                            
                            if existing_id:
                                logger.info(f"{node.label} node with content '{node_content}' already exists with id {existing_id}, skipping creation")
                                # Update the node's id to match the existing one for relationship creation
                                node.properties['id'] = existing_id
                                continue

                            logger.info(f"Creating new {node.label} node with content '{node_content}'")
                            await self._create_node(session, node, common_metadata)

                        except Exception as e:
                            logger.error(f"Error processing node {node.label} with content '{node_content}': {e}")
                            raise

                    # Create relationships after nodes
                    for rel in relationship_objects:
                        await self._create_relationship(session, rel, common_metadata)

            

            except Exception as e:
                logger.error(f"Error in store_llm_generated_graph: {e}")
                logger.error("Full traceback:", exc_info=True)
                raise

    async def _create_node(self, session, node: Node, common_metadata: dict):
        """Helper method to create a single node"""
        # Merge node properties with common metadata
        props = {k: v for k, v in node.properties.items() if v is not None}
        if 'metadata' in props and isinstance(props['metadata'], dict):
            props['metadata'] = json.dumps(props['metadata'])
        
        # Add common metadata
        props.update({k: v for k, v in common_metadata.items() if v is not None})

        # Create node with label and properties
        # Fix: Remove NodeLabel. prefix from the label
        # Get the label value from the NodeLabel enum if it's an enum, otherwise use as is
        label = node.label.value if isinstance(node.label, NodeLabel) else node.label
        query = (
            f"CREATE (n:{label} $props) "
            "RETURN n"
        )
        try:
            result = await session.run(query, props=props)
            await result.consume()
            logger.info(f"Created node with label {label} and id {props.get('id')}")
        except Exception as e:
            logger.error(f"Error creating node: {e}")
            raise
    
    async def _index_exists_async(self, session, index_name: str) -> bool:
        """Check if an index exists (async version)"""
        result = await session.run("SHOW INDEXES WHERE name = $name", 
                                 {"name": index_name})
        records = await result.values()
        return len(records) > 0

    async def initialize_indexes_async(self):
        """Creates necessary indexes if they don't exist (async version)"""
        driver = await self.async_neo_conn.get_driver()
        if driver is None:
            logger.warning("No Neo4j connection available, skipping index initialization")
            return
            
        try:
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, skipping index initialization")
                return
            
            # Base indexes for ID and name
            indexes = {
                "person_id_idx": "CREATE INDEX person_id_idx IF NOT EXISTS FOR (n:Person) ON (n.id)",
                "company_id_idx": "CREATE INDEX company_id_idx IF NOT EXISTS FOR (n:Company) ON (n.id)",
                "customer_id_idx": "CREATE INDEX customer_id_idx IF NOT EXISTS FOR (n:Customer) ON (n.id)",
                "project_id_idx": "CREATE INDEX project_id_idx IF NOT EXISTS FOR (n:Project) ON (n.id)",
                "memory_id_idx": "CREATE INDEX memory_id_idx IF NOT EXISTS FOR (n:Memory) ON (n.id)",
                "task_id_idx": "CREATE INDEX task_id_idx IF NOT EXISTS FOR (n:Task) ON (n.id)",
                "insight_id_idx": "CREATE INDEX insight_id_idx IF NOT EXISTS FOR (n:Insight) ON (n.id)",
                "opportunity_id_idx": "CREATE INDEX opportunity_id_idx IF NOT EXISTS FOR (n:Opportunity) ON (n.id)",
                "code_id_idx": "CREATE INDEX code_id_idx IF NOT EXISTS FOR (n:Code) ON (n.id)",
                "meeting_id_idx": "CREATE INDEX meeting_id_idx IF NOT EXISTS FOR (n:Meeting) ON (n.id)",
                
                # Content-based index for Memory nodes
                "memory_content_idx": "CREATE INDEX memory_content_idx IF NOT EXISTS FOR (n:Memory) ON (n.content)",
                
                # Name-based indexes for entity nodes
                "person_name_idx": "CREATE INDEX person_name_idx IF NOT EXISTS FOR (n:Person) ON (n.name)",
                "company_name_idx": "CREATE INDEX company_name_idx IF NOT EXISTS FOR (n:Company) ON (n.name)",
                "customer_name_idx": "CREATE INDEX customer_name_idx IF NOT EXISTS FOR (n:Customer) ON (n.name)",
                "project_name_idx": "CREATE INDEX project_name_idx IF NOT EXISTS FOR (n:Project) ON (n.name)"
            }
            
            # Add access control indexes for each node type from NodeLabel enum
            for node_type in NodeLabel:
                node_label = node_type.value
                node_name = node_label.lower()
                
                # User ID index
                indexes[f"{node_name}_user_id_idx"] = f"CREATE INDEX {node_name}_user_id_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.user_id)"
                
                # Workspace access index
                indexes[f"{node_name}_workspace_access_idx"] = f"CREATE INDEX {node_name}_workspace_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.workspace_read_access)"
                
                # User access index
                indexes[f"{node_name}_user_access_idx"] = f"CREATE INDEX {node_name}_user_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.user_read_access)"
                
                # Role access index
                indexes[f"{node_name}_role_access_idx"] = f"CREATE INDEX {node_name}_role_access_idx IF NOT EXISTS FOR (n:{node_label}) ON (n.role_read_access)"
            
            #async with driver.session() as session:
            async with self.async_neo_conn.get_session() as session:
                for index_name, create_query in indexes.items():
                    if not await self._index_exists_async(session, index_name):
                        await session.run(create_query)
                        logger.info(f"Created new index: {index_name}")
                    else:
                        logger.info(f"Index already exists: {index_name}")
                        
        except Exception as e:
            logger.error(f"Error managing indexes: {e}")
            raise

    async def _create_relationship(self, session, relationship: Relationship, common_metadata: dict):
        """Helper method to create a single relationship"""
        try:
            # Create a properties dictionary for the relationship
            props = {
                "created_at": datetime.utcnow().isoformat(),
                **common_metadata
            }
            
             # Get relationship type
            relationship_type = relationship.type.value if isinstance(relationship.type, RelationshipType) else relationship.type
            
            # Get source and target information
            source = relationship.source
            target = relationship.target
            
            # Get source and target IDs and labels
            source_id = source.id if hasattr(source, 'id') else source['id']
            target_id = target.id if hasattr(target, 'id') else target['id']
            
            # Get labels from source and target
            source_label = source.label if hasattr(source, 'label') else source['label']
            target_label = target.label if hasattr(target, 'label') else target['label']
            
            # Convert labels to their string values if they're enums
            source_label = source_label.value if isinstance(source_label, NodeLabel) else source_label
            target_label = target_label.value if isinstance(target_label, NodeLabel) else target_label
            
            # Optimized query using MATCH with relationship pattern
            query = f"""
            MATCH (source:{source_label} {{id: $source_id}})
            MATCH (target:{target_label} {{id: $target_id}})
            WHERE source.id = $source_id AND target.id = $target_id
            MERGE (source)-[r:{relationship_type}]->(target)
            ON CREATE SET r += $props
            RETURN r
            """
            
            result = await session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                props=props
            )
            await result.consume()
            logger.info(f"Created relationship {relationship_type} from {source_id} to {target_id}")
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise

   
   
    @staticmethod
    def get_memory_graph_schema():
        """
        Returns a fixed memory graph schema from structured outputs.
        """
        from models.structured_outputs import MemoryGraphSchema
        return MemoryGraphSchema.get_fixed_json_schema()
    
    @staticmethod
    def get_node_schema():
        """
        Returns a fixed memory graph schema from structured outputs.
        """
        from models.structured_outputs import Node
        return Node.get_fixed_json_schema()
    
    @staticmethod
    def get_relationship_schema():
        """
        Returns a fixed memory graph schema from structured outputs.
        """
        from models.structured_outputs import Relationship
        return Relationship.get_fixed_json_schema()

    @staticmethod
    def get_memory_only_schema():
        """
        Returns a fixed memory graph schema from structured outputs.
        """
        from models.structured_outputs import MemoryGraphSchema
        return MemoryGraphSchema.get_memory_only_schema()


    async def add_memory_item_async(
        self, 
        memory_item: MemoryItem,
        relationships_json: dict,
        sessionToken: str,
        user_id: str,
        background_tasks: BackgroundTasks,
        add_to_pinecone: bool = True,
        workspace_id: Optional[str] = None,
        skip_background_processing: bool = False,
        user_workspace_ids: Optional[List[str]] = None
    ) -> List[ParseStoredMemory]:  # Updated return type
        """
        Async version of add_memory_item that quickly stores the memory and processes relationships in the background.
        
        Args:
            memory_item (MemoryItem): The memory item to add
            relationships_json (dict): JSON describing relationships
            sessionToken (str): Session token for authentication
            user_id (str): User ID
            background_tasks (BackgroundTasks): FastAPI background tasks handler
            add_to_pinecone (bool): Whether to add to Pinecone
            workspace_id (Optional[str]): Workspace ID
            skip_background_processing (bool): If True, skips adding background tasks for processing
            
        Returns:
            List[ParseStoredMemory]: List of stored memory items from Parse Server
        """
        try:
            # Store memory item in instance dictionary
            if memory_item and hasattr(memory_item, 'id'):
                self.memory_items[memory_item.id] = memory_item
            
            # If no workspace_id provided, try to get it from selected workspace follower
            if not workspace_id:
                async with AsyncClient() as client:
                    workspace_id = await User.get_selected_workspace_id_async(user_id, sessionToken)
                    if workspace_id:
                        logger.info(f"Using selected workspace ID: {workspace_id}")
                        memory_item.metadata['workspace_id'] = workspace_id
                    else:
                        logger.warning("No workspace_id provided and no selected workspace found")

            # Initialize variables for the return values
            added_item_properties: ParseStoredMemory = None
            memory_item_obj: MemoryItem = None

            if add_to_pinecone:
                # Add memory item without relationships first and wait for the result
                added_item_properties, memory_list = await self.add_memory_item_without_relationships(sessionToken, memory_item, user_workspace_ids)
                memory_item_list: List[MemoryItem] = memory_list
                logger.info(f'memory_item_list: {memory_item_list}')
                added_item_properties: List[ParseStoredMemory] = added_item_properties
                logger.info(f'added_item_properties: {added_item_properties}')

                if added_item_properties and memory_item_list:
                    # Since we're now dealing with single items, get the first item
                    added_item = added_item_properties[0]  # This is a ParseStoredMemory object
                    logger.info(f'Added item properties memoryId: {added_item.memoryId}')
                    memory_item_obj = memory_item_list[0]
                    logger.info(f'Memory item obj id: {memory_item_obj.id}')
                    
                    # Update the memory_item using proper attribute access
                    memory_item_obj.objectId = added_item.objectId
                    memory_item_obj.createdAt = added_item.createdAt
                    
                    # Get memoryChunkIds safely
                    try:
                        # First try to get metadata as dict
                        metadata = added_item.metadata
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        
                        # Get chunk IDs from metadata or use empty list as fallback
                        chunk_ids = metadata.get('memoryChunkIds', []) if isinstance(metadata, dict) else []
                        memory_item_obj.memoryChunkIds = chunk_ids
                        logger.info(f'Updated memoryChunkIds: {chunk_ids}')
                    except Exception as e:
                        logger.error(f'Error processing memoryChunkIds: {e}')
                        memory_item_obj.memoryChunkIds = []
                    
                    logger.info(f'Memory item obj: {memory_item_obj}')

                    # Convert memory_item to a fully serializable dictionary
                    memory_item_dict = memory_item_to_dict(memory_item_obj)
                    logger.info(f'Converted memory_item to dict: {memory_item_dict}')

                    logger.info(f'skip_background_processing: {skip_background_processing}')
                                    
                    # Skip background processing tasks when skip_background_processing is True
                    if not skip_background_processing:
                        logger.info(f'Adding background task for processing')
                        # Add background task for processing
                        background_tasks.add_task(
                            self.process_memory_item_async,
                            session_token=sessionToken,
                            memory_dict=memory_item_dict,
                            relationships_json=relationships_json,
                            workspace_id=workspace_id,
                            user_id=user_id
                        )

                        # Process relationships in background
                        if memory_item_obj.context and len(memory_item_obj.context) > 0:
                            logger.info(f'Context for memory item exists: {memory_item_obj.context}')
                            background_tasks.add_task(
                                self.update_memory_item_with_relationships,
                                memory_item_obj,
                                relationships_json,
                                workspace_id,
                                user_id
                            )

                return added_item_properties

            return []  # Return empty list if not adding to pinecone

        except Exception as e:
            logger.error(f"Error in add_memory_item_async: {e}")
            logger.error("Full traceback:", exc_info=True)
            return []  # Return empty list on error

    async def check_neo4j_health(self):
        """Check Neo4j connection health"""
        try:
            await self.ensure_async_connection()
            if self.async_neo_conn.fallback_mode:
                logger.warning("Neo4j in fallback mode, skipping health check")
                return False
            driver = await self.async_neo_conn.get_driver()
            async with driver.session() as session:
                start = time.time()
                result = await session.run("RETURN 1")
                await result.consume()
                latency = time.time() - start         
                logger.info(f"Neo4j health check successful. Latency: {latency:.2f}s")
                # Reset fallback mode if health check is successful
                if self.async_neo_conn.fallback_mode:
                    logger.info("Resetting fallback mode as connection is healthy")
                    self.async_neo_conn.fallback_mode = False
                return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {str(e)}")
            self.async_neo_conn.fallback_mode = True
            return False


