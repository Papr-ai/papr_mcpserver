import logging
import os
from logging.handlers import RotatingFileHandler
import sys

def setup_logging():
    """Set up logging configuration with error handling"""
    try:
        # Set up logging with more detailed format
        log_dir = 'logs'
        
        # Ensure logs directory exists and is writable
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, mode=0o755)
            except Exception as e:
                print(f"Error creating logs directory: {e}", file=sys.stderr)
                # Fallback to current directory if logs directory can't be created
                log_dir = '.'
        
        # Create a rotating file handler
        log_file = os.path.join(log_dir, 'debug.log')
        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        except Exception as e:
            print(f"Error setting up file handler: {e}", file=sys.stderr)
            file_handler = None

        # Create a stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        # Configure root logger
        handlers = [stream_handler]
        if file_handler:
            handlers.append(file_handler)
            
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=handlers
        )
        
        # Log that logging is set up
        logger = logging.getLogger(__name__)
        logger.info("Logging system initialized")
        
    except Exception as e:
        print(f"Error setting up logging: {e}", file=sys.stderr)
        # Fallback to basic console logging
        logging.basicConfig(level=logging.DEBUG)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)

# Initialize logging when module is imported
setup_logging() 