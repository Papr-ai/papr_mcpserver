from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from services.logging_config import get_logger

logger = get_logger(__name__)

class ParseStoredMemory(BaseModel):
    """Model for a memory item stored in Parse Server"""
    objectId: str
    createdAt: datetime
    updatedAt: datetime
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    type: str
    sourceUrl: str = ""
    conversationId: str = ""
    memoryId: str
    topics: List[str] = []
    steps: List[str] = []
    current_step: Optional[str] = None
    memoryChunkIds: List[str] = []
    user: Dict[str, Any]  # ParseUserPointer
    workspace: Optional[Dict[str, Any]] = None  # ParsePointer
    post: Optional[Dict[str, Any]] = None  # ParsePointer
    postMessage: Optional[Dict[str, Any]] = None  # ParsePointer
    matchingChunkIds: Optional[List[str]] = None
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    upload_id: Optional[str] = None
    extract_mode: Optional[str] = None
    fileStructures: Optional[str] = None
    
    # Additional fields from API response
    ACL: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    sourceType: str = ""
    title: Optional[str] = None
    location: str = "online"
    emojiTags: List[str] = Field(default_factory=list)
    hierarchicalStructures: str = ""
    file_url: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[int] = None

    @field_validator('metadata', mode='before')
    @classmethod
    def validate_metadata(cls, v: Any) -> Dict[str, Any]:
        """Ensure metadata is a dictionary"""
        if v is None:
            return {}
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        if isinstance(v, dict):
            return v
        return {}

    @field_validator('context', mode='before')
    @classmethod
    def validate_context(cls, v: Any) -> Dict[str, Any]:
        """Convert string context to dictionary"""
        if v is None:
            return {}
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list) and parsed:
                    # If it's a list with items, take the first item as context
                    return parsed[0] if isinstance(parsed[0], dict) else {}
                elif isinstance(parsed, dict):
                    return parsed
                return {}
            except json.JSONDecodeError:
                return {}
        if isinstance(v, dict):
            return v
        return {}

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        extra='forbid'
    )

class NeoNode(BaseModel):
    """Model for a Neo4j node"""
    label: str
    properties: Dict[str, Any]

class RelatedMemoryResult(BaseModel):
    """Model for related memory search results"""
    memory_items: List[ParseStoredMemory]
    neo_nodes: Optional[List[NeoNode]] = None

class GetMemoryResponse(BaseModel):
    """Response model for get_memory API endpoint"""
    data: RelatedMemoryResult
    success: bool = True
    error: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class AddMemoryResponse(BaseModel):
    """Response model for add_memory API endpoint"""
    code: int = Field(default=200)
    data: List[Dict[str, Any]]  # List[AddMemoryItem]
    status: str = Field(default="success")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        extra='forbid'
    )

class BatchMemoryResponse(BaseModel):
    """Response model for batch memory operations"""
    successful: List[AddMemoryResponse]
    errors: List[Dict[str, Any]]  # List[BatchMemoryError]
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

class DeleteMemoryResponse(BaseModel):
    """Response model for delete_memory API endpoint"""
    message: Optional[str] = None
    error: Optional[str] = None
    memoryId: str
    objectId: str
    status: Dict[str, bool]  # DeletionStatus
    code: str
    status_code: int

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class UpdateMemoryResponse(BaseModel):
    """Response model for update_memory API endpoint"""
    message: str = "Memory item(s) updated"
    memory_items: Optional[List[Dict[str, Any]]] = None  # List[UpdateMemoryItem]
    success: bool = True
    error: Optional[str] = None
    status: Dict[str, bool] = Field(default_factory=lambda: {"pinecone": False, "neo4j": False, "parse": False})
    code: int = 200

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        extra='forbid'
    ) 