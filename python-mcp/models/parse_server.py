from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal, TypedDict, Union
from datetime import datetime
from enum import Enum
import json
from fastapi import HTTPException
from services.logging_config import get_logger
from models.model_validators import (
    ParseStoredMemory,
    GetMemoryResponse,
    AddMemoryResponse,
    BatchMemoryResponse,
    DeleteMemoryResponse,
    UpdateMemoryResponse
)

logger = get_logger(__name__)



class ParsePointer(BaseModel):
    """A pointer to a Parse object"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")  # Using Field with alias
    className: str

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        alias_generator=None
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        # Ensure the type field is output as __type
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

class AddMemoryItem(BaseModel):
    """Response model for a single memory item in add_memory response"""
    memoryId: str
    createdAt: datetime
    objectId: str
    memoryChunkIds: List[str] = Field(default_factory=list)
    
    @field_validator('memoryChunkIds', mode='before')
    @classmethod
    def validate_memory_chunk_ids(cls, v) -> List[str]:
        logger.debug(f"Validating memoryChunkIds input: {v} of type {type(v)}")
        if v is None:
            logger.debug("memoryChunkIds is None, returning empty list")
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    logger.debug(f"Parsed memoryChunkIds from JSON string: {parsed}")
                    return [str(x) for x in parsed if x]
            except json.JSONDecodeError:
                # If it's a comma-separated string
                if ',' in v:
                    chunks = [x.strip() for x in v.split(',')]
                    logger.debug(f"Split memoryChunkIds from comma-separated string: {chunks}")
                    return [x for x in chunks if x]
                # If it's a single value
                if v.strip():
                    logger.debug(f"Single memoryChunkId from string: {[v.strip()]}")
                    return [v.strip()]
        if isinstance(v, list):
            logger.debug(f"memoryChunkIds is already a list: {v}")
            return [str(x) for x in v if x]
        logger.warning(f"Unexpected memoryChunkIds type: {type(v)}, returning empty list")
        return []

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        extra='forbid'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to handle datetime serialization"""
        d = super().model_dump(*args, **kwargs)
        if isinstance(d.get('createdAt'), datetime):
            d['createdAt'] = d['createdAt'].isoformat()
        return d

class AddMemoryResponse(BaseModel):
    """Response model for add_memory API endpoint"""
    code: int = Field(default=200)
    data: List[AddMemoryItem]
    status: str = Field(default="success")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        extra='forbid'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure nested datetime serialization"""
        d = super().model_dump(*args, **kwargs)
        if d.get('data'):
            for item in d['data']:
                if isinstance(item.get('createdAt'), datetime):
                    item['createdAt'] = item['createdAt'].isoformat()
        return d

class ErrorDetail(BaseModel):
    code: int
    detail: str

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class MemoryErrorResponse(BaseModel):
    status_code: int
    success: bool = False
    error: str
    data: Optional[Any] = None

class BatchMemoryError(BaseModel):
    index: int
    error: str

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class BatchMemoryResponse(BaseModel):
    successful: List[AddMemoryResponse]
    errors: List[BatchMemoryError]
    total_processed: int
    total_successful: int
    total_failed: int
    total_content_size: int
    total_storage_size: int

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        extra='forbid'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure nested datetime serialization"""
        d = super().model_dump(*args, **kwargs)
        if d.get('successful'):
            for response in d['successful']:
                if response.get('data'):
                    for item in response['data']:
                        if isinstance(item.get('createdAt'), datetime):
                            item['createdAt'] = item['createdAt'].isoformat()
        return d

class ParseUserPointer(BaseModel):
    """A pointer to a Parse User object that can also handle expanded user objects"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["_User"] = "_User"

    # Add optional fields that might come from Parse
    displayName: Optional[str] = None
    fullname: Optional[str] = None
    profileimage: Optional[str] = None
    title: Optional[str] = None
    isOnline: Optional[bool] = None
    profileimage: Optional[Union[str, Dict[str, Any]]] = None
    companies: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        alias_generator=None
    )

    @field_validator('profileimage')
    @classmethod
    def validate_profileimage(cls, v):
        if v is None:
            return None
        # If it's already a string (URL), return it
        if isinstance(v, str):
            return v
        # If it's a Parse File object, return the URL
        if isinstance(v, dict):
            return v.get('url')

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        # Ensure the type field is output as __type
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

class ParseStoredMemory(BaseModel):
    objectId: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Dict[str, Dict[str, bool]]
    content: str
    metadata: Optional[Union[str, Dict[str, Any], None]] = None
    sourceType: str = "papr"
    context: str = None
    title: Optional[str] = None
    location: Optional[str] = None
    emojiTags: List[str] = Field(default_factory=list)
    hierarchicalStructures: str = ""
    type: str
    sourceUrl: str = ""
    conversationId: str = ""
    memoryId: str
    topics: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    current_step: Optional[str] = None
    memoryChunkIds: List[str] = Field(default_factory=list)
    user: ParseUserPointer
    workspace: Optional[ParsePointer] = None
    post: Optional[ParsePointer] = None
    postMessage: Optional[ParsePointer] = None
    matchingChunkIds: Optional[List[str]] = None

    # Document specific fields
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    upload_id: Optional[str] = None
    extract_mode: Optional[str] = None
    file_url: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[str] = None

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        logger.debug(f"Validating metadata: {v}")
        if v is None:
            logger.debug("Metadata is None, returning empty dict")
            return {}
        return v

    model_config = ConfigDict(
        from_attributes=True,  # Allows conversion from ORM objects
        validate_assignment=True,  # Validate during assignment
        populate_by_name=True,  # Allow population by field name as well as alias
        str_strip_whitespace=True,  # Strip whitespace from strings
        extra='allow'  # Allow extra attributes
    )

    def without_metadata(self) -> 'ParseStoredMemory':
        """Create a copy of the memory item without metadata field."""
        return ParseStoredMemory.model_validate(
            self.model_dump(exclude={'metadata'})
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'ParseStoredMemory':
        logger.debug(f"Creating ParseStoredMemory from dict: {data}")
        
        # If metadata is a string, try to parse it
        if isinstance(data.get('metadata'), str):
            try:
                metadata = json.loads(data['metadata'])
                logger.debug(f"Parsed metadata from string: {metadata}")
                # Check if memoryChunkIds exists in metadata
                if 'memoryChunkIds' in metadata:
                    logger.debug(f"Found memoryChunkIds in metadata: {metadata['memoryChunkIds']}")
                    # Make sure to use these IDs
                    data['memoryChunkIds'] = metadata['memoryChunkIds']
            except json.JSONDecodeError:
                logger.error(f"Failed to parse metadata string: {data['metadata']}")
        
        logger.debug(f"Final data before conversion: {data}")
        instance = cls(**data)
        logger.debug(f"Created instance with memoryChunkIds: {instance.memoryChunkIds}")
        return instance

    @field_validator('memoryChunkIds', mode='before')
    @classmethod
    def validate_memory_chunk_ids(cls, v: Any) -> List[str]:
        """Ensure memoryChunkIds is always a list of clean strings"""
        logger.debug(f"Validating memoryChunkIds: {v}")
        if v is None:
            logger.debug("memoryChunkIds is None, returning empty list")
            return []
        
        # If it's already a list, clean each item
        if isinstance(v, list):
            logger.debug(f"Cleaning memoryChunkIds list: {v}")
            # Clean each item by removing quotes and brackets
            return [str(x).strip().strip("'[]\"") for x in v if x]
            
        # If it's a string that looks like a list
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    logger.debug(f"Parsed memoryChunkIds from JSON: {parsed}")
                    return [str(x).strip().strip("'[]\"") for x in parsed if x]
            except json.JSONDecodeError:
                # If it's a string representation of a list
                if v.startswith('[') and v.endswith(']'):
                    # Remove outer brackets and split
                    items = v[1:-1].split(',')
                    logger.debug(f"Split memoryChunkIds from string list: {items}")
                    return [x.strip().strip("'[]\"") for x in items if x.strip()]
                # If it's a comma-separated string
                if ',' in v:
                    items = v.split(',')
                    logger.debug(f"Split memoryChunkIds from comma-separated string: {items}")
                    return [x.strip().strip("'[]\"") for x in items if x.strip()]
                # Single value
                if v.strip():
                    logger.debug(f"Single memoryChunkId: {[v.strip()]}")
                    return [v.strip().strip("'[]\"")]
        
        logger.warning(f"Unexpected memoryChunkIds format: {v}")
        return []
    
    @classmethod
    def from_parse_response(cls, response_data: Dict[str, Any]) -> 'ParseStoredMemory':
        """Create ParseStoredMemory from Parse Server response"""
        # Extract base fields
        base_data = {
            'objectId': response_data['objectId'],
            'createdAt': response_data['createdAt'],
            'updatedAt': response_data.get('updatedAt'),
            'ACL': response_data.get('ACL', {}),
            'content': response_data['content'],
            'metadata': response_data.get('metadata', {}),
            'sourceType': response_data.get('sourceType', 'papr'),
            'context': response_data.get('context'),
            'title': response_data.get('title'),
            'location': response_data.get('location'),
            'type': response_data.get('type', 'TextMemoryItem'),
            'topics': response_data.get('topics', []),
            'memoryChunkIds': response_data.get('memoryChunkIds', []),
            'user': response_data.get('user')
        }

        # Add document-specific fields if this is a DocumentMemoryItem
        if base_data['type'] == 'DocumentMemoryItem':
            doc_fields = {
                'page_number': response_data.get('page_number'),
                'total_pages': response_data.get('total_pages'),
                'upload_id': response_data.get('upload_id'),
                'extract_mode': response_data.get('extract_mode'),
                'file_url': response_data.get('file_url'),
                'filename': response_data.get('filename'),
                'page': response_data.get('page')
            }
            base_data.update(doc_fields)

        return cls(**base_data)
    
class MemoryRetrievalResult(TypedDict):
    results: List[ParseStoredMemory]
    missing_memory_ids: List[str]


class UseCaseMetrics(TypedDict):
    usecase_token_count_input: int
    usecase_token_count_output: int
    usecase_total_cost: float

class UseCaseResponse(TypedDict):
    data: Dict[str, List[Dict[str, Any]]]  # Contains 'goals' and 'use_cases' lists
    metrics: UseCaseMetrics
    refusal: Optional[str]
   
class RelatedMemoriesMetrics(TypedDict):
    related_memories_token_count_input: int
    related_memories_token_count_output: int
    related_memories_total_cost: float

class RelatedMemoriesSuccess(TypedDict):
    data: List[ParseStoredMemory]
    generated_queries: List[str]
    metrics: RelatedMemoriesMetrics

class RelatedMemoriesError(TypedDict):
    error: str
    generated_queries: List[str]
    metrics: RelatedMemoriesMetrics

class DeletionStatus(BaseModel):
    pinecone: bool = False
    neo4j: bool = False
    parse: bool = False

class DeleteMemoryResponse(BaseModel):
    message: Optional[str] = None
    error: Optional[str] = None
    memoryId: str
    objectId: str
    status: DeletionStatus
    code: str
    status_code: int  # 200 for SUCCESS, 207 for PARTIAL_DELETE

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class DeleteMemoryResult(TypedDict):
    response: DeleteMemoryResponse
    status_code: int

# Define types for Pinecone response
class PineconeMatch(TypedDict):
    id: str
    score: float
    metadata: dict
    values: Optional[List[float]]

class PineconeQueryResponse(TypedDict):
    matches: List[PineconeMatch]
    namespace: str

# Error response model
class ErrorResponse(BaseModel):
    error: str

class InteractionLimits(TypedDict):
    mini: int
    premium: int

class TierLimits(TypedDict):
    pro: InteractionLimits
    business_plus: InteractionLimits
    free_trial: InteractionLimits

class SystemUpdateStatus(BaseModel):
    """Status of update operation for each system"""
    pinecone: bool = False
    neo4j: bool = False
    parse: bool = False

class UpdateMemoryItem(BaseModel):
    """Model for a single updated memory item"""
    objectId: str
    memoryId: str
    content: Optional[str] = "" 
    updatedAt: datetime
    memoryChunkIds: Optional[List[str]] = []

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid',
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        }
    )

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        if isinstance(d.get('updatedAt'), datetime):
            d['updatedAt'] = d['updatedAt'].isoformat()
        return d

class UpdateMemoryResponse(BaseModel):
    """Response model for update_memory API endpoint"""
    message: str = "Memory item(s) updated"
    memory_items: Optional[List[UpdateMemoryItem]] = None
    success: bool = True
    error: Optional[str] = None
    status: SystemUpdateStatus = Field(default_factory=SystemUpdateStatus)
    code: int = 200

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid',
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        }
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to handle datetime serialization"""
        d = super().model_dump(*args, **kwargs)
        if d.get('memory_items'):
            for item in d['memory_items']:
                if isinstance(item.get('updatedAt'), datetime):
                    item['updatedAt'] = item['updatedAt'].isoformat()
        return d

    @classmethod
    def error_response(cls, error_message: str, code: int = 500, status: Optional[SystemUpdateStatus] = None) -> 'UpdateMemoryResponse':
        """Create an error response"""
        return cls(
            message="Update failed",
            success=False,
            error=error_message,
            status=status or SystemUpdateStatus(),
            code=code
        )

    @classmethod
    def success_response(cls, items: List[UpdateMemoryItem], status: SystemUpdateStatus) -> 'UpdateMemoryResponse':
        """Create a success response"""
        return cls(
            message="Memory item(s) updated",
            memory_items=items,
            success=True,
            status=status,
            code=200
        )
    

class MemoryParseServer(BaseModel):
    """Model that exactly matches the Parse Server Memory class schema"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Dict[str, Dict[str, bool]]
    title: Optional[str] = None
    content: Optional[str] = None
    file: Optional[Dict[str, Any]] = None
    user: ParsePointer
    workspace: Optional[ParsePointer] = None
    topics: Optional[List[str]] = Field(default_factory=list)
    location: Optional[str] = None
    emojiTags: Optional[List[str]] = Field(default_factory=list)
    emotionTags: Optional[List[str]] = Field(default_factory=list)
    hierarchicalStructures: Optional[str] = None
    type: Optional[str] = None
    sourceUrl: Optional[str] = None
    conversationId: Optional[str] = None
    memoryId: Optional[str] = None
    imageURL: Optional[str] = None
    sourceType: Optional[str] = Field(default="papr")
    context: Optional[str] = None
    goals: Optional[Dict[str, Any]] = None  # For Relation type
    usecases: Optional[Dict[str, Any]] = None  # For Relation type
    node: Optional[ParsePointer] = None
    relationship_json: Optional[List[Any]] = Field(default_factory=list)
    node_name: Optional[str] = None
    postMessage: Optional[ParsePointer] = None
    current_step: Optional[str] = None
    steps: Optional[List[str]] = Field(default_factory=list)
    post: Optional[ParsePointer] = None
    totalCost: Optional[float] = None
    tokenSize: Optional[int] = None
    storageSize: Optional[int] = None
    usecaseGenerationCost: Optional[float] = None
    schemaGenerationCost: Optional[float] = None
    relatedMemoriesCost: Optional[float] = None
    nodeDefinitionCost: Optional[float] = None
    bigbirdEmbeddingCost: Optional[float] = None
    sentenceBertCost: Optional[float] = None
    totalProcessingCost: Optional[float] = None
    memoryIds: Optional[List[str]] = Field(default_factory=list)
    memoryChunkIds: Optional[List[str]] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # New DocumentMemoryItem specific fields
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    upload_id: Optional[str] = None
    extract_mode: Optional[str] = None
    file_url: Optional[str] = None  # Parse Server file URL
    filename: Optional[str] = None
    page: Optional[str] = None  # Format: "X of Y"

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure proper handling of ACL, pointer fields, and DocumentMemoryItem"""
        data = super().model_dump(*args, **kwargs)
        
        # Helper function to transform pointer fields
        def transform_pointer(pointer_data):
            if pointer_data and isinstance(pointer_data, dict):
                if 'type' in pointer_data:
                    pointer_data['__type'] = pointer_data.pop('type')
            return pointer_data
        
        # Transform all pointer fields
        pointer_fields = ['user', 'workspace', 'node', 'postMessage', 'post']
        for field in pointer_fields:
            if field in data:
                data[field] = transform_pointer(data[field])
        
        # Ensure ACL is properly formatted
        if 'ACL' in data and isinstance(data['ACL'], dict):
            # If ACL is a dict of single characters, reconstruct it properly
            if any(len(k) == 1 for k in data['ACL'].keys()):
                reconstructed_acl = {}
                current_key = []
                
                for k, v in sorted(data['ACL'].items()):
                    if k.startswith('role:'):
                        reconstructed_acl[k] = v
                    else:
                        current_key.append(k)
                        if k in ['[', ']']:
                            key = ''.join(current_key)
                            if key not in reconstructed_acl:
                                reconstructed_acl[key] = v
                            current_key = []
                
                data['ACL'] = reconstructed_acl
        
        # Handle DocumentMemoryItem fields
        if data.get('type') != 'DocumentMemoryItem':
            for field in ['extract_mode']:
                data.pop(field, None)
        
        return data

    @field_validator('memoryIds', 'topics', 'emojiTags', 'steps', 'relationship_json')
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                # Try to parse if it's a JSON string
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # If it's a comma-separated string or string representation of a list
                if v.startswith('[') and v.endswith(']'):
                    # Handle string representation of a list
                    try:
                        # Remove brackets and split by comma
                        items = v[1:-1].split(',')
                        # Clean up each item
                        return [item.strip().strip("'\"") for item in items if item.strip()]
                    except Exception:
                        return []
                return [x.strip() for x in v.split(',') if x.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('memoryChunkIds')
    @classmethod
    def validate_memory_chunk_ids(cls, v: Any) -> List[str]:
        """Ensure memoryChunkIds is always a list of strings"""
        if v is None:
            return []
        
        # If it's already a list, clean it up
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
            
        # If it's a string that looks like a list
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except json.JSONDecodeError:
                # If it's a comma-separated string
                if ',' in v:
                    return [x.strip() for x in v.split(',') if x.strip()]
                # Single value
                if v.strip():
                    return [v.strip()]
        
        return []

    @field_validator('steps')
    @classmethod
    def format_steps(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                # Try to parse if it's a JSON string
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    # Clean up any string representations of lists
                    return [item.strip("[]'\" ") for item in parsed]
                return []
            except json.JSONDecodeError:
                # If it's a string representation of a list
                if v.startswith('[') and v.endswith(']'):
                    try:
                        # Remove brackets and split by comma
                        items = v[1:-1].split(',')
                        # Clean up each item
                        return [item.strip().strip("[]'\" ") for item in items if item.strip()]
                    except Exception:
                        return []
                # If it's a comma-separated string
                return [step.strip().strip("[]'\" ") for step in v.split(',') if step.strip()]
        if isinstance(v, list):
            # Clean up any string representations of lists in the array
            return [item.strip("[]'\" ") if isinstance(item, str) else item for item in v]
        return []

    @field_validator('emojiTags')
    @classmethod
    def format_emoji_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('emotionTags')
    @classmethod
    def format_emotion_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('topics')
    @classmethod
    def format_topics(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [topic.strip() for topic in v.split(',') if topic.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('hierarchicalStructures')
    @classmethod
    def format_hierarchical_structures(cls, v):
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        if isinstance(v, dict):
            return json.dumps(v)
        return str(v)

    @field_validator('*', mode='before')
    @classmethod
    def handle_none(cls, v):
        return v if v is not None else None


class MemoryParseServerUpdate(BaseModel):
    """Model for updating existing Memory items in Parse Server"""
    ACL: Optional[Dict[str, Dict[str, bool]]] = None
    content: Optional[str] = None
    sourceType: Optional[str] = Field(default="papr")
    context: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    emojiTags: Optional[List[str]] = None
    emotionTags: Optional[List[str]] = None
    hierarchicalStructures: Optional[str] = None
    type: Optional[str] = None
    sourceUrl: Optional[str] = None
    conversationId: Optional[str] = None
    topics: Optional[List[str]] = None
    steps: Optional[List[str]] = None
    current_step: Optional[str] = None
    memoryChunkIds: Optional[List[str]] = Field(default_factory=list)
    relationship_json: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    memoryIds: Optional[List[str]] = Field(default_factory=list)
    totalCost: Optional[float] = None
    tokenSize: Optional[int] = None
    storageSize: Optional[int] = None
    usecaseGenerationCost: Optional[float] = None
    schemaGenerationCost: Optional[float] = None
    relatedMemoriesCost: Optional[float] = None
    nodeDefinitionCost: Optional[float] = None
    bigbirdEmbeddingCost: Optional[float] = None
    sentenceBertCost: Optional[float] = None
    totalProcessingCost: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    upload_id: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[str] = None
    file_url: Optional[str] = None
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )

    @field_validator('memoryChunkIds')
    @classmethod
    def validate_memory_chunk_ids(cls, v: Optional[Any]) -> List[str]:
        """Ensure memoryChunkIds is a list of strings"""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []

    @field_validator('emojiTags', 'emotionTags', 'topics', 'steps')
    @classmethod
    def ensure_list(cls, v: Optional[Any]) -> List[str]:
        """Ensure fields are always lists"""
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in v.split(',') if x.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
        return []

    @field_validator('hierarchicalStructures')
    @classmethod
    def format_hierarchical_structures(cls, v: Optional[Any]) -> str:
        """Format hierarchical structures as string"""
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        if isinstance(v, dict):
            return json.dumps(v)
        return str(v)
    

class DocumentUploadStatusType(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"
    QUEUED = "queued"  # Optional: if we want to indicate queued state
    CANCELLED = "cancelled"  # Optional: if we want to support cancellation


class DocumentUploadResponse(BaseModel):
    message: str
    upload_id: str
    status: DocumentUploadStatusType
    memory_items: List[AddMemoryItem]

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        },
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )


class DocumentUploadStatus(BaseModel):
    status: DocumentUploadStatusType
    progress: float  # 0.0 to 1.0 for percentage
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    current_filename: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        },
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class DocumentUploadStatusResponse(BaseModel):
    """Response model for document upload status from Parse Server"""
    objectId: str
    status: DocumentUploadStatusType
    filename: Optional[str] = None
    progress: float
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    current_filename: Optional[str] = None
    error: Optional[str] = None
    upload_id: str
    user: ParseUserPointer

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        },
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )