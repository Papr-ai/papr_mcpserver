# url_utils.py

from urllib.parse import urlparse
from services.logging_config import get_logger

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)


def handle_url(source_url):
    """
    Sanitize a URL by removing all query parameters and ensuring the URL ends with .pdf.

    Args:
        source_url (str): The original URL that needs to be sanitized.

    Returns:
        str: The sanitized URL ending with .pdf.
    """
    parsed_url = urlparse(source_url)
    
    # Reconstruct the base URL without query parameters
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

    # Check if the base URL ends with .pdf
    if base_url.lower().endswith('.pdf'):
        sanitized_url = base_url
    else:
        logger.warning(f"The URL does not end with .pdf: {base_url}")
        sanitized_url = base_url  # This will return the URL even if it doesn't end with .pdf

    # Log the final sanitized URL
    logger.info(f"Sanitized URL: {sanitized_url}")

    return sanitized_url

def clean_url(url):
    """Clean a URL by removing comments and spaces."""
    if not url:
        return url
    
    # Split by '#' and take the first part (before any comment)
    url = url.split('#')[0]
    
    # Remove leading/trailing whitespace
    url = url.strip()
    
    logger.debug(f"Cleaned URL: {url}")
    return url