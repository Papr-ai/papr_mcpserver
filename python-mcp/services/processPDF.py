import os
import fitz  # PyMuPDF
from langchain_community.document_loaders import UnstructuredHTMLLoader, TextLoader, PyPDFLoader
import spacy
from collections import Counter
from services.logging_config import get_logger
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from background_tasks.pdf_processing import add_page_to_memory_task
from fastapi import BackgroundTasks
from models.parse_server import AddMemoryResponse, AddMemoryItem
import json
import uuid
from os import environ as env
from memory.memory_graph import MemoryGraph
import httpx
from services.document_status import update_processing_status
from models.parse_server import DocumentUploadStatusType
from services.memory_management import get_post_file_info
from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Initialize spaCy for NLP tasks
nlp = spacy.load("en_core_web_sm")

from dotenv import find_dotenv, load_dotenv

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


def process_pdf(pdf_file, filename, context=None):
    """
    Original synchronous PDF processing function.
    Used by data parsers and tests.
    """
    # Check the file type and select the appropriate loader
    if filename.lower().endswith('.pdf'):
        loader_class = PyPDFLoader
    elif filename.lower().endswith('.html'):
        loader_class = UnstructuredHTMLLoader
    elif filename.lower().endswith('.txt'):
        loader_class = TextLoader
    else:
        logger.error(f"Unsupported file type: {filename}")
        return  # Ignore unsupported file types without raising an error

    save_path = os.path.join('/tmp', filename)
    with open(save_path, 'wb') as file:
        file.write(pdf_file.read())
    logger.info(f"Processing file: {filename}")

    try:
        loader = loader_class(save_path)  # Use the appropriate loader
        pages = loader.load()
        total_pages = len(pages)  # Get total number of pages
        page_number = 1  # Initialize page_number

        for page in pages:
            content = page.page_content

            # Use the provided context or default to an empty list
            context_array = context if context is not None else []
            
            # Extracting topics from content
            doc = nlp(content)
            topics = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            topic_counts = Counter(topics)
            most_common_topics = topic_counts.most_common(5)
            topics_only = [topic for topic, _ in most_common_topics]  # Extract only topics, not counts

            # Constructing metadata with topics and file/page information
            metadata = {
                "topics": ', '.join(topics_only),  # Convert list of topics to a string
                "file": filename,
                "page": f"{page_number} of {total_pages}"
            }
            # Yielding page content, context, and topics
            yield {
                "content": content,
                "context": context_array,
                "metadata": metadata
            }
            page_number += 1

        os.remove(save_path)
        logger.info(f"Successfully processed and removed the file: {filename}")

    except Exception as e:
        os.remove(save_path)
        logger.error(f"An error occurred while processing the file {filename}: {e}", exc_info=True)
        raise e

async def save_uploaded_file(file_content: bytes, filename: str) -> Tuple[str, str]:
    """
    Save an uploaded file and return its path and detected file type.
    """
    import magic
    
    # Save file to temp location
    temp_file_path = os.path.join('/tmp', filename)
    with open(temp_file_path, 'wb') as f:
        f.write(file_content)
    
    # Check file type
    file_type = magic.from_file(temp_file_path, mime=True)
    logger.info(f"Detected file type for {filename}: {file_type}")
    
    return temp_file_path, file_type

async def extract_text_from_pdf_async(pdf_path: str, file_url: str, extract_mode: str = 'text') -> List[Dict[str, Any]]:
    """
    Asynchronously extract text from PDF using PyMuPDF with different extraction modes.
    
    Args:
        pdf_path: Path to the PDF file
        file_url: URL of the file in Parse Server
        extract_mode: 'text' for plain text, 'blocks' for structured blocks, 
                     'dict' for detailed information including bounds
    """
    pages_content = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            if extract_mode == 'blocks':
                # Get text in blocks - better for maintaining structure
                blocks = page.get_text("blocks")
                content = "\n".join([block[4] for block in blocks])  # block[4] contains the text
                
                # Extract additional block information with enhanced type detection
                blocks_info = []
                previous_block = None
                
                for block in blocks:
                    text = block[4]
                    bounds = block[:4]  # x0, y0, x1, y1
                    font_info = block[5] if len(block) > 5 else None
                    
                    # Initialize block info
                    block_info = {
                        "text": text,
                        "bounds": bounds,
                        "block_type": "text"  # default type
                    }
                    
                    # Detect block type based on characteristics
                    text_stripped = text.strip()
                    
                    # Header detection
                    if text_stripped and (
                        # Check if text is short and followed by larger vertical gap
                        (len(text_stripped.split()) <= 10 and
                         previous_block and
                         abs(previous_block["bounds"][3] - bounds[1]) > 20) or
                        # Check if text starts with common header patterns
                        any(text_stripped.lower().startswith(pattern) for pattern in 
                            ['chapter', 'section', 'part ', 'appendix'])
                    ):
                        block_info["block_type"] = "header"
                    
                    # List item detection
                    elif text_stripped and (
                        # Bullet points
                        text_stripped.startswith(('•', '-', '∙', '○', '●')) or
                        # Numbered lists
                        (len(text_stripped) > 2 and
                         text_stripped[0].isdigit() and
                         text_stripped[1] in [')', '.', '-'])
                    ):
                        block_info["block_type"] = "list_item"
                    
                    # Table cell detection (simplified)
                    elif previous_block and (
                        # Check for aligned blocks with similar y-coordinates
                        abs(bounds[1] - previous_block["bounds"][1]) < 5 and
                        abs(bounds[3] - previous_block["bounds"][3]) < 5
                    ):
                        block_info["block_type"] = "table_cell"
                    
                    # Paragraph detection
                    elif len(text_stripped.split()) > 10:
                        block_info["block_type"] = "paragraph"
                    
                    # Caption detection
                    elif text_stripped.lower().startswith(('figure', 'fig.', 'table', 'image')):
                        block_info["block_type"] = "caption"
                    
                    blocks_info.append(block_info)
                    previous_block = {
                        "text": text,
                        "bounds": bounds
                    }
                
            elif extract_mode == 'dict':
                # Get detailed dictionary including text, images, and formatting
                content = page.get_text("dict")
                blocks_info = content["blocks"]  # Contains detailed block information
                content = page.get_text()  # Still get plain text for processing
                
            else:  # Default to 'text'
                content = page.get_text()
                blocks_info = []

            # Extract images if available
            images = []
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    images.append({
                        "image_data": base_image["image"],
                        "image_type": base_image["ext"],
                        "index": img_index
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract image: {e}")

            # Create page content with proper metadata
            page_data = {
                "content": content,
                "page_number": page_num + 1,  # Correct page numbering
                "metadata": {
                    "page": f"{page_num + 1} of {total_pages}",
                    "file_url": file_url,
                    "source_url": file_url,
                    "blocks": blocks_info if extract_mode in ['blocks', 'dict'] else None,
                    "images": images
                }
            }
            
            pages_content.append(page_data)
            await asyncio.sleep(0)
            
        return pages_content
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

async def process_single_page(
    page_data: Dict,
    total_pages: int,
    filename: str,
    upload_id: str,
    context: List[Dict],
    user_id: str,
    session_token: str,
    workspace_id: str,
    background_tasks: BackgroundTasks,
    extract_mode: str,
    memory_graph: MemoryGraph,
    client_type: str = 'papr_plugin',
    memory_objectId: str = None,
    user_workspace_ids: Optional[List[str]] = None,
    post_objectId: Optional[str] = None  # New parameter
) -> Dict[str, Any]:
    """Process a single page from a PDF document."""
    try:
        page_number = page_data.get("page_number", 1)
        content = page_data["content"]
        # Get file_url and source_url from page_data metadata
        file_url = page_data.get("metadata", {}).get("file_url")
        source_url = page_data.get("metadata", {}).get("source_url")
        
        logger.info(f"Processing page {page_number} of {total_pages} from {filename}")

        # Extract topics using spaCy
        doc = nlp(content)
        topics = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        topic_counts = Counter(topics)
        most_common_topics = topic_counts.most_common(5)
        topics_only = [topic for topic, _ in most_common_topics]

        # Create enriched metadata for the page
        metadata = {
            "filename": filename,
            "page_number": page_number,
            "total_pages": total_pages,
            "upload_id": upload_id,
            "extract_mode": extract_mode,
            "topics": ', '.join(topics_only),
            "page": f"{page_number} of {total_pages}",
            "workspace_id": workspace_id,
            "file_url": file_url,  # Add file_url
            "sourceUrl": source_url  # Add source_url
        }

        processed_page = {
            "content": content,
            "context": context if context is not None else [],
            "metadata": metadata,
            "type": "DocumentMemoryItem",
            "sourceType": "papr"
        }

        logger.info(f"Attempting to add page {page_number} to memory with metadata: {metadata}")

        try:
            # Use add_page_to_memory_task with correct parameters and enriched metadata
            memory_items = await add_page_to_memory_task(
                processed_page=processed_page,
                metadata=metadata,
                user_id=user_id,
                session_token=session_token,
                workspace_id=workspace_id,
                memory_graph=memory_graph,
                background_tasks=background_tasks,
                client_type=client_type,
                user_workspace_ids=user_workspace_ids
            )
            
            if memory_items and len(memory_items) > 0:
                logger.info(f"Successfully added page {page_number} to memory. Memory ID: {memory_items[0].memoryId}")
                return memory_items[0]
            else:
                raise RuntimeError(f"Failed to process page {page_number} - no memory item returned")

        except ValueError as ve:
            # Handle validation errors (invalid token, empty content)
            logger.error(f"Validation error processing page {page_number}: {str(ve)}")
            await update_processing_status(
                upload_id=upload_id,
                filename=filename,
                current_page=page_number,
                total_pages=total_pages,
                status=DocumentUploadStatusType.FAILED,
                error=str(ve),
                objectId=memory_objectId,
                post_objectId=post_objectId,
                file_url=file_url
            )
            raise

        except RuntimeError as re:
            # Handle memory addition failures
            logger.error(f"Runtime error processing page {page_number}: {str(re)}")
            await update_processing_status(
                upload_id=upload_id,
                filename=filename,
                current_page=page_number,
                total_pages=total_pages,
                status=DocumentUploadStatusType.FAILED,
                error=str(re),
                objectId=memory_objectId,
                post_objectId=post_objectId,
                file_url=file_url
            )
            raise

        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error processing page {page_number}: {str(e)}", exc_info=True)
            await update_processing_status(
                upload_id=upload_id,
                filename=filename,
                current_page=page_number,
                total_pages=total_pages,
                status=DocumentUploadStatusType.FAILED,
                error=str(e),
                objectId=memory_objectId,
                post_objectId=post_objectId,
                file_url=file_url
            )
            raise

    except Exception as e:
        logger.error(f"Error processing page {page_number}: {str(e)}", exc_info=True)
        await update_processing_status(
                upload_id=upload_id,
                filename=filename,
                current_page=page_number,
                total_pages=total_pages,
                status=DocumentUploadStatusType.FAILED,
                error=str(e),
                objectId=memory_objectId,
                post_objectId=post_objectId,
                file_url=file_url
            )
        raise

async def process_pdf_in_background(
    file_url: str,  
    filename: str,
    context: List[Dict] = None,
    upload_id: str = None,
    update_status_callback = None,
    user_id: str = None,
    session_token: str = None,
    memory_objectId: str = None,
    workspace_id: str = None,
    background_tasks: BackgroundTasks = None,
    extract_mode: str = 'blocks',
    batch_size: int = 10,
    memory_graph: MemoryGraph = None,
    client_type: str = 'papr_plugin',
    user_workspace_ids: Optional[List[str]] = None,
    post_objectId: Optional[str] = None
) -> AddMemoryResponse:
    """Process document files asynchronously with optimized batch processing"""
    processed_pages = []
    failed_pages = []
    temp_file = None
    
    try:
        # If post_objectId is provided, fetch file info from Post
        if post_objectId:
            try:
                file_url = await get_post_file_info(post_objectId)
                logger.info(f"File URL: {file_url}")
            except ValueError as ve:
                logger.error(f"Error getting file from Post: {str(ve)}")
                if update_status_callback:
                    await update_status_callback(
                        upload_id=upload_id,
                        filename=filename,
                        status=DocumentUploadStatusType.FAILED,
                        error=str(ve),
                        objectId=memory_objectId,
                        post_objectId=post_objectId,
                        file_url=file_url
                    )
                raise

        # Download file from Parse Server temporarily
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = os.path.join('/tmp', filename)
            with open(temp_file, 'wb') as f:
                f.write(response.content)

        # Select appropriate loader based on file extension
        if filename.lower().endswith('.pdf'):
            pages_content = await extract_text_from_pdf_async(temp_file, file_url, extract_mode)
            logger.info(f"pages_content from extract_text_from_pdf_async: {pages_content}")
        elif filename.lower().endswith('.html'):
            loader = UnstructuredHTMLLoader(temp_file)
            pages = loader.load()
            pages_content = [{
                "content": page.page_content,
                "page_number": idx + 1,
                "metadata": {
                    "file_url": file_url,
                    "source_url": file_url,
                    "filename": filename
                }
            } for idx, page in enumerate(pages)]
        elif filename.lower().endswith('.txt'):
            loader = TextLoader(temp_file)
            pages = loader.load()
            pages_content = [{
                "content": page.page_content,
                "page_number": idx + 1,
                "metadata": {
                    "file_url": file_url,
                    "source_url": file_url,
                    "filename": filename
                }
            } for idx, page in enumerate(pages)]
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        total_pages = len(pages_content)
        logger.info(f"Total pages: {total_pages}")
        
        # Create all tasks at once
        tasks = [
            process_single_page(
                page_data=page_data,
                total_pages=total_pages,
                filename=filename,
                upload_id=upload_id,
                context=context,
                user_id=user_id,
                session_token=session_token,
                workspace_id=workspace_id,
                background_tasks=background_tasks,
                extract_mode=extract_mode,
                memory_graph=memory_graph,
                client_type=client_type,
                memory_objectId=memory_objectId,
                user_workspace_ids=user_workspace_ids,
                post_objectId=post_objectId 
            )
            for page_data in pages_content
        ]

        # Process in batches using asyncio.gather
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Handle results, separating successes and failures
            for j, result in enumerate(batch_results):
                current_page = i + j + 1
                if isinstance(result, Exception):
                    logger.error(f"Error processing page {current_page}: {str(result)}")
                    failed_pages.append(current_page)
                elif result is None:
                    logger.error(f"Failed to process page {current_page} - returned None")
                    failed_pages.append(current_page)
                else:
                    processed_pages.append(result)

            if update_status_callback:
                await update_status_callback(
                    upload_id=upload_id,
                    filename=filename,
                    current_page=min(i + batch_size, total_pages),
                    total_pages=total_pages,
                    status=DocumentUploadStatusType.PROCESSING,
                    objectId=memory_objectId,
                    post_objectId=post_objectId,
                    file_url=file_url
                )

        # Final status update
        final_status = "completed" if not failed_pages else "completed_with_errors"
        error_message = f"Failed to process pages: {failed_pages}" if failed_pages else None
        
        if update_status_callback:
            final_status = (DocumentUploadStatusType.COMPLETED 
                        if not failed_pages 
                        else DocumentUploadStatusType.FAILED)
            error_message = f"Failed to process pages: {failed_pages}" if failed_pages else None
            
            await update_status_callback(
                upload_id=upload_id,
                filename=filename,
                current_page=total_pages,
                total_pages=total_pages,
                status=final_status,
                error=error_message,
                objectId=memory_objectId,
                post_objectId=post_objectId,
                file_url=file_url
            )


        return AddMemoryResponse(
            code=200,
            status="success",
            data=processed_pages
        )

    except Exception as e:
        logger.error(f"An error occurred while processing the file {filename}: {e}", exc_info=True)
        if update_status_callback:
            await update_status_callback(
                upload_id=upload_id,
                filename=filename,
                status=DocumentUploadStatusType.FAILED,
                error=str(e),
                objectId=memory_objectId,
                post_objectId=post_objectId,
                file_url=file_url
            )
        return AddMemoryResponse(
            code=500,
            status="error",
            data=[],
            error=str(e)
        )
    finally:
        # Clean up temporary file if it exists
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info(f"Cleaned up temporary file: {temp_file}")

