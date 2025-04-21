from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from services.logging_config import get_logger

logger = get_logger(__name__)

class ParseStoredMemory(BaseModel):
    """Model for memory items stored in Parse Server"""
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
    user: Dict[str, Any]  # ParseUserPointer
    workspace: Optional[Dict[str, Any]] = None  # ParsePointer
    post: Optional[Dict[str, Any]] = None  # ParsePointer
    postMessage: Optional[Dict[str, Any]] = None  # ParsePointer
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
        from_attributes=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        extra='forbid'
    )

    @field_validator('memoryChunkIds', mode='before')
    @classmethod
    def validate_memory_chunk_ids(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x]
            except json.JSONDecodeError:
                if ',' in v:
                    return [x.strip() for x in v.split(',') if x.strip()]
                if v.strip():
                    return [v.strip()]
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []

class GetMemoryResponse(BaseModel):
    """Response model for get_memory API endpoint"""
    data: Dict[str, Any]  # RelatedMemoryResult
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