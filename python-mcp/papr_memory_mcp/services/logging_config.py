# Copyright 2024 Papr AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
