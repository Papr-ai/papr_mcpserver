import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
import httpx
from paprmcp import (
    get_memory,
    add_memory,
    add_document,
    add_memory_batch,
    add_memory_batch_with_details,
    update_memory,
    delete_memory,
    get_document_status
)

# Test fixtures
@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, {
        "MEMERY_SERVER_URL": "https://memory.papr.ai",
        "PAPR_API_KEY": "test_api_key"
    }):
        yield

@pytest.fixture
def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value = AsyncMock()
        yield mock_client

@pytest.mark.asyncio
async def test_get_memory(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "memory_items": [],
            "neo_nodes": [],
            "neo_context": None,
            "neo_query": None
        },
        "success": True,
        "error": None
    }
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response

    # Test the function
    result = await get_memory("test query")
    assert result["success"] is True
    assert result["data"]["memory_items"] == []
    assert result["data"]["neo_nodes"] == []
    assert result["error"] is None

@pytest.mark.asyncio
async def test_add_memory(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{
            "memoryId": "test_id",
            "createdAt": "2024-01-01T00:00:00Z",
            "objectId": "test_object_id",
            "memoryChunkIds": []
        }],
        "status": "success",
        "code": 200
    }
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response

    # Test the function
    result = await add_memory("test content")
    assert result["status"] == "success"
    assert result["code"] == 200
    assert len(result["data"]) == 1

@pytest.mark.asyncio
async def test_add_document(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{
            "memoryId": "test_id",
            "createdAt": "2024-01-01T00:00:00Z",
            "objectId": "test_object_id",
            "memoryChunkIds": []
        }],
        "status": "success",
        "code": 200
    }
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response

    # Test the function
    result = await add_document("test document content")
    assert result["status"] == "success"
    assert result["code"] == 200
    assert len(result["data"]) == 1

@pytest.mark.asyncio
async def test_add_memory_batch(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "successful": [{
            "data": [{
                "memoryId": "test_id",
                "createdAt": "2024-01-01T00:00:00Z",
                "objectId": "test_object_id",
                "memoryChunkIds": []
            }],
            "status": "success",
            "code": 200
        }],
        "errors": [],
        "total_processed": 1,
        "total_successful": 1,
        "total_failed": 0,
        "total_content_size": 100,
        "total_storage_size": 200
    }
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response

    # Test the function
    result = await add_memory_batch([{"content": "test content"}])
    assert result["total_successful"] == 1
    assert result["total_failed"] == 0

@pytest.mark.asyncio
async def test_add_memory_batch_with_details(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "successful": [{
            "data": [{
                "memoryId": "test_id",
                "createdAt": "2024-01-01T00:00:00Z",
                "objectId": "test_object_id",
                "memoryChunkIds": []
            }],
            "status": "success",
            "code": 200
        }],
        "errors": [],
        "total_processed": 1,
        "total_successful": 1,
        "total_failed": 0,
        "total_content_size": 100,
        "total_storage_size": 200
    }
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response

    # Test the function
    result = await add_memory_batch_with_details(
        [{"content": "test content"}],
        "tenant/subtenant",
        "connector",
        "stream"
    )
    assert result["total_successful"] == 1
    assert result["total_failed"] == 0

@pytest.mark.asyncio
async def test_update_memory(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": "Memory item(s) updated",
        "memory_items": [{
            "objectId": "test_id",
            "memoryId": "test_memory_id",
            "content": "updated content",
            "updatedAt": "2024-01-01T00:00:00Z"
        }],
        "success": True,
        "error": None,
        "status": {
            "pinecone": True,
            "neo4j": True,
            "parse": True
        },
        "code": 200
    }
    mock_httpx_client.return_value.__aenter__.return_value.put.return_value = mock_response

    # Test the function
    result = await update_memory("test_id", content="updated content")
    assert result["success"] is True
    assert result["code"] == 200

@pytest.mark.asyncio
async def test_delete_memory(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": "Memory deleted successfully",
        "error": None,
        "memoryId": "test_id",
        "objectId": "test_object_id",
        "status": {
            "pinecone": True,
            "neo4j": True,
            "parse": True
        },
        "code": "200",
        "status_code": 200
    }
    mock_httpx_client.return_value.__aenter__.return_value.delete.return_value = mock_response

    # Test the function
    result = await delete_memory("test_id")
    assert result["code"] == "200"
    assert result["memoryId"] == "test_id"
    assert result["status_code"] == 200

@pytest.mark.asyncio
async def test_get_document_status(mock_env_vars, mock_httpx_client):
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "processing_status": "completed",
        "chunk_count": 5,
        "error": None,
        "last_updated": "2024-01-01T00:00:00Z"
    }
    mock_httpx_client.return_value.__aenter__.return_value.get.return_value = mock_response

    # Test the function
    result = await get_document_status("test_id")
    assert result["processing_status"] == "completed"
    assert result["chunk_count"] == 5

@pytest.mark.asyncio
async def test_error_handling(mock_env_vars, mock_httpx_client):
    # Mock HTTP error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found",
        request=MagicMock(),
        response=MagicMock()
    )
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response

    # Test error handling
    with pytest.raises(httpx.HTTPStatusError):
        await get_memory("test query")

    # Test missing API key
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            await add_memory("test content")

@pytest.mark.asyncio
async def test_missing_api_key(mock_env_vars):
    # Test missing API key
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            await get_memory("test query") 