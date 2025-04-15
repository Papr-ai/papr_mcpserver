from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator, RootModel
from typing import Dict, Any, List, Optional, Union
from models.structured_outputs import Node
from datetime import datetime
from memory.memory_item import MemoryItem
import json
from models.structured_outputs import (
    NodeLabel, MemoryProperties, PersonProperties, CompanyProperties, 
    ProjectProperties, TaskProperties, InsightProperties, MeetingProperties, 
    OpportunityProperties, CodeProperties
)
from services.logging_config import get_logger
from models.parse_server import ParseStoredMemory
from pydantic import validator
import logging

# Create a logger instance for this module
logger = get_logger(__name__)  # Will use 'memory_service.memory.memory_graph' as the logger name

class MemoryNodeProperties(BaseModel):
    """Properties specific to Memory nodes"""
    id: str
    type: str
    content: str
    memoryChunkIds: List[str]
    user_id: str
    workspace_id: Optional[str] = None
    pageId: Optional[str] = None
    title: Optional[str] = None
    topics: Optional[List[str]] = Field(default_factory=list)
    emotion_tags: Optional[List[str]] = Field(default_factory=list)
    emoji_tags: Optional[List[str]] = Field(default_factory=list)
    hierarchical_structures: Optional[str] = Field(default="")
    conversationId: Optional[str] = None
    sourceType: Optional[str] = None
    sourceUrl: Optional[str] = None
    user_read_access: List[str] = Field(default_factory=list)
    user_write_access: List[str] = Field(default_factory=list)
    workspace_read_access: List[str] = Field(default_factory=list)
    workspace_write_access: List[str] = Field(default_factory=list)
    role_read_access: List[str] = Field(default_factory=list)
    role_write_access: List[str] = Field(default_factory=list)
    createdAt: Optional[str] = None

def memory_item_to_node(memory_item: 'MemoryItem', chunk_ids: List[str]) -> Node:
    """Convert a MemoryItem to a Node object for Neo4j storage"""
    # Extract metadata
    metadata = (json.loads(memory_item.metadata) 
               if isinstance(memory_item.metadata, str) 
               else memory_item.metadata)
    
    # Helper function to convert comma-separated string to list
    def string_to_list(value: Optional[Union[str, List[str]]]) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return value
        return [item.strip() for item in value.split(',')]

    # Create properties using Pydantic model for validation
    properties = MemoryNodeProperties(
        id=str(memory_item.id),
        type=memory_item.type,
        content=memory_item.content,
        memoryChunkIds=chunk_ids,
        user_id=metadata.get('user_id', ''),
        workspace_id=metadata.get('workspace_id'),
        pageId=metadata.get('pageId'),
        title=metadata.get('title'),
        # Convert string fields to lists
        topics=string_to_list(metadata.get('topics')),
        emotion_tags=string_to_list(metadata.get('emotion_tags', 
                                  metadata.get('emotionTags',
                                  metadata.get('emotion tags', '')))),
        emoji_tags=string_to_list(metadata.get('emoji_tags', 
                                  metadata.get('emojiTags',
                                  metadata.get('emoji tags', '')))),
        hierarchical_structures=(metadata.get('hierarchical_structures') or 
                               metadata.get('hierarchicalStructures') or 
                               metadata.get('hierarchical structures') or 
                               ''),
        conversationId=metadata.get('conversationId'),
        sourceType=metadata.get('sourceType'),
        sourceUrl=metadata.get('sourceUrl'),
        user_read_access=metadata.get('user_read_access', []),
        user_write_access=metadata.get('user_write_access', []),
        workspace_read_access=metadata.get('workspace_read_access', []),
        workspace_write_access=metadata.get('workspace_write_access', []),
        role_read_access=metadata.get('role_read_access', []),
        role_write_access=metadata.get('role_write_access', []),
        createdAt=metadata.get('createdAt') or datetime.utcnow().isoformat()
    ).model_dump(exclude_none=True)

    # Create Node object
    return Node(
        label="Memory",
        properties=properties
    )

class NeoBaseProperties(BaseModel):
    """Base properties that all Neo4j nodes should inherit from"""
    model_config = ConfigDict(
        extra='forbid'
    )
    
    # Optional ACL properties
    user_read_access: Optional[List[str]] = Field(default_factory=list)
    user_write_access: Optional[List[str]] = Field(default_factory=list)
    workspace_read_access: Optional[List[str]] = Field(default_factory=list)
    workspace_write_access: Optional[List[str]] = Field(default_factory=list)
    role_read_access: Optional[List[str]] = Field(default_factory=list)
    role_write_access: Optional[List[str]] = Field(default_factory=list)
    
    # Optional metadata fields
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    sourceType: Optional[str] = None
    sourceUrl: Optional[str] = None
    pageId: Optional[str] = None
    conversationId: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

class NeoMemoryNode(MemoryProperties, NeoBaseProperties):
    """Memory node with all Neo4j properties"""
    title: Optional[str] = None
    emoji_tags: Optional[List[str]] = Field(default_factory=list)
    hierarchical_structures: Optional[str] = Field(default="")

    pass

class NeoPersonNode(PersonProperties, NeoBaseProperties):
    """Person node with all Neo4j properties"""
    pass

class NeoCompanyNode(CompanyProperties, NeoBaseProperties):
    """Company node with all Neo4j properties"""
    pass

class NeoProjectNode(ProjectProperties, NeoBaseProperties):
    """Project node with all Neo4j properties"""
    pass

class NeoTaskNode(TaskProperties, NeoBaseProperties):
    """Task node with all Neo4j properties"""
    pass

class NeoInsightNode(InsightProperties, NeoBaseProperties):
    """Insight node with all Neo4j properties"""
    pass

class NeoMeetingNode(MeetingProperties, NeoBaseProperties):
    """Meeting node with all Neo4j properties"""
    pass

class NeoOpportunityNode(OpportunityProperties, NeoBaseProperties):
    """Opportunity node with all Neo4j properties"""
    pass

class NeoCodeNode(CodeProperties, NeoBaseProperties):
    """Code node with all Neo4j properties"""
    pass

class NeoNode(BaseModel):
    """Generic Neo4j node that combines label and type-specific properties"""
    label: NodeLabel
    properties: Union[
        NeoMemoryNode,
        NeoPersonNode,
        NeoCompanyNode,
        NeoProjectNode,
        NeoTaskNode,
        NeoInsightNode,
        NeoMeetingNode,
        NeoOpportunityNode,
        NeoCodeNode
    ]


class MemorySourceLocation(BaseModel):
    """
    Tracks presence of a memory item in different storage systems.
    """
    in_pinecone: bool = Field(default=False, description="Memory exists in Pinecone")
    in_bigbird: bool = Field(default=False, description="Memory exists in BigBird")
    in_neo: bool = Field(default=False, description="Memory exists in Neo4j")

class MemoryIDSourceLocation(BaseModel):
    memory_id: str
    source_location: MemorySourceLocation

class MemorySourceInfo(BaseModel):
    memory_id_source_location: List[MemoryIDSourceLocation]

    
class RelatedMemoryResult(BaseModel):
    """Return type for find_related_memory_items_async"""
    memory_items: List[ParseStoredMemory]
    neo_nodes: List[NeoNode]
    neo_context: Optional[str] = None
    neo_query: Optional[str] = None
    memory_source_info: Optional[MemorySourceInfo] = None

    def log_summary(self) -> None:
        """Log a summary of the results"""
        logger.info(f"Found {len(self.memory_items)} memory items")
        logger.info(f"Found {len(self.neo_nodes)} neo nodes")
        logger.info(f"Found {len(self.memory_source_info.memory_id_source_location)} source locations")
        if self.neo_nodes:
            logger.info("Sample of neo nodes (first 3):")
            for i, node in enumerate(self.neo_nodes[:3]):
                logger.info(f"Node {i + 1}:")
                logger.info(f"  Label: {node.label}")
                logger.info(f"  Properties: {json.dumps(node.properties.model_dump(), indent=2)}")
        logger.info(f"Neo context: {self.neo_context}")
        logger.info(f"Neo query: {self.neo_query}")


class GetMemoryResponse(BaseModel):
    data: RelatedMemoryResult
    success: bool = True
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [],
                "success": True,
                "error": None
            }
        }
    )

class NodeConverter:
    @staticmethod
    def _parse_access_lists(node_dict: dict) -> dict:
        """Convert string representations of lists to actual lists"""
        access_fields = [
            'user_read_access',
            'user_write_access',
            'workspace_read_access',
            'workspace_write_access',
            'role_read_access',
            'role_write_access'
        ]
        
        parsed_dict = node_dict.copy()
        for field in access_fields:
            if field in parsed_dict:
                try:
                    # Handle string representation of list
                    if isinstance(parsed_dict[field], str):
                        # Remove quotes and brackets, split by comma
                        value = parsed_dict[field].strip('[]\'\"')
                        if value:
                            # Split by comma and clean up each item
                            parsed_dict[field] = [item.strip().strip('\'\"') for item in value.split(',')]
                        else:
                            parsed_dict[field] = []
                except Exception as e:
                    logger.warning(f"Error parsing {field}: {str(e)}")
                    parsed_dict[field] = []
                    
        return parsed_dict

    @staticmethod
    def _parse_string_to_list(value: Union[str, List[str], None]) -> List[str]:
        """Convert string or list to proper list format"""
        if not value:
            return []
        if isinstance(value, list):
            return value
        # Must be string at this point since we've defined Union[str, List[str], None]
        return [item.strip() for item in value.split(',')]

    @staticmethod
    def convert_to_neo_node(node_dict: dict, primary_label: str) -> Optional[NeoNode]:
        """
        Convert a dictionary of node properties to the appropriate NeoNode object
        
        Args:
            node_dict (dict): Dictionary containing node properties
            primary_label (str): Primary label of the node ('Task', 'Memory', etc.)
            
        Returns:
            Optional[NeoNode]: Converted node or None if conversion fails
        """
        try:
             # Parse access control lists first
            parsed_dict = NodeConverter._parse_access_lists(node_dict)

            # Handle topics and other list fields
            if 'topics' in parsed_dict:
                parsed_dict['topics'] = NodeConverter._parse_string_to_list(parsed_dict['topics'])
            if 'emotion_tags' in parsed_dict:
                parsed_dict['emotion_tags'] = NodeConverter._parse_string_to_list(parsed_dict.get('emotionTags', parsed_dict.get('emotion_tags', '')))
            if 'emoji_tags' in parsed_dict:
                parsed_dict['emoji_tags'] = NodeConverter._parse_string_to_list(parsed_dict.get('emojiTags', parsed_dict.get('emoji_tags', '')))
                
            # Rename hierarchicalStructures to hierarchical_structures
            if 'hierarchicalStructures' in parsed_dict:
                parsed_dict['hierarchical_structures'] = parsed_dict.pop('hierarchicalStructures')

            if primary_label == 'Task':
                props = NeoTaskNode(
                    id=parsed_dict['id'],
                    title=parsed_dict.get('title', ''),
                    description=parsed_dict.get('description', ''),
                    status=parsed_dict.get('status', ''),
                    type=parsed_dict.get('type', ''),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Task, properties=props)

            elif primary_label == 'Memory':
                props = NeoMemoryNode(
                    id=parsed_dict['id'],
                    content=parsed_dict.get('content', ''),
                    type=parsed_dict.get('type', ''),
                    topics=parsed_dict.get('topics', []),
                    emotion_tags=parsed_dict.get('emotion_tags', []),
                    steps=parsed_dict.get('steps', []),
                    current_step=parsed_dict.get('current_step', ''),
                    title=parsed_dict.get('title'),
                    emoji_tags=parsed_dict.get('emoji_tags', []),
                    hierarchical_structures=parsed_dict.get('hierarchical_structures', ''),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Memory, properties=props)

            elif primary_label == 'Person':
                props = NeoPersonNode(
                    id=parsed_dict['id'],
                    name=parsed_dict.get('name', ''),
                    role=parsed_dict.get('role', ''),
                    description=parsed_dict.get('description', ''),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Person, properties=props)

            elif primary_label == 'Company':
                props = NeoCompanyNode(
                    id=parsed_dict['id'],
                    name=parsed_dict.get('name', ''),
                    description=parsed_dict.get('description', ''),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Company, properties=props)

            elif primary_label == 'Project':
                props = NeoProjectNode(
                    id=parsed_dict['id'],
                    name=parsed_dict.get('name', ''),
                    type=parsed_dict.get('type', ''),
                    description=parsed_dict.get('description', ''),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Project, properties=props)

            elif primary_label == 'Insight':
                props = NeoInsightNode(
                    id=parsed_dict['id'],
                    title=parsed_dict.get('title', ''),
                    description=parsed_dict.get('description', ''),
                    source=parsed_dict.get('source', ''),
                    type=parsed_dict.get('type', 'other'),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Insight, properties=props)

            elif primary_label == 'Meeting':
                props = NeoMeetingNode(
                    id=parsed_dict['id'],
                    title=parsed_dict.get('title', ''),
                    agenda=parsed_dict.get('agenda', ''),
                    type=parsed_dict.get('type', ''),
                    participants=parsed_dict.get('participants', []),
                    date=parsed_dict.get('date', ''),
                    time=parsed_dict.get('time', ''),
                    summary=parsed_dict.get('summary', ''),
                    outcome=parsed_dict.get('outcome', ''),
                    action_items=parsed_dict.get('action_items', []),
                   **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Meeting, properties=props)

            elif primary_label == 'Opportunity':
                props = NeoOpportunityNode(
                    id=parsed_dict['id'],
                    title=parsed_dict.get('title', ''),
                    description=parsed_dict.get('description', ''),
                    value=float(parsed_dict.get('value', 0)),
                    stage=parsed_dict.get('stage', 'prospect'),
                    close_date=parsed_dict.get('close_date', ''),
                    probability=float(parsed_dict.get('probability', 0)),
                    next_steps=parsed_dict.get('next_steps', []),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Opportunity, properties=props)

            elif primary_label == 'Code':
                props = NeoCodeNode(
                    id=parsed_dict['id'],
                    name=parsed_dict.get('name', ''),
                    language=parsed_dict.get('language', ''),
                    author=parsed_dict.get('author', ''),
                    **NodeConverter._get_base_properties(parsed_dict)
                )
                return NeoNode(label=NodeLabel.Code, properties=props)

            else:
                logger.warning(f"Unhandled node type: {primary_label}")
                return None

        except ValidationError as e:
            logger.error(f"Validation error for {primary_label} node:")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Failed properties: {parsed_dict}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {primary_label} node: {str(e)}")
            logger.error(f"Node data: {parsed_dict}")
            return None
        
    @staticmethod
    def _get_base_properties(node_dict: dict) -> dict:
        """Extract common base properties from node dictionary"""
        return {
            'user_id': node_dict.get('user_id'),
            'workspace_id': node_dict.get('workspace_id'),
            'sourceType': node_dict.get('sourceType'),
            'sourceUrl': node_dict.get('sourceUrl'),
            'pageId': node_dict.get('pageId'),
            'conversationId': node_dict.get('conversationId'),
            'createdAt': node_dict.get('createdAt'),
            'updatedAt': node_dict.get('updatedAt'),
            'user_read_access': node_dict.get('user_read_access', []),
            'user_write_access': node_dict.get('user_write_access', []),
            'workspace_read_access': node_dict.get('workspace_read_access', []),
            'workspace_write_access': node_dict.get('workspace_write_access', []),
            'role_read_access': node_dict.get('role_read_access', []),
            'role_write_access': node_dict.get('role_write_access', [])
        }

