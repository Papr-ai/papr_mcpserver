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
import datetime
from os import environ as env
from dotenv import find_dotenv, load_dotenv

# Load environment variables immediately
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

class LoggerSingleton:
    _instance = None
    _loggers = {}  # Dictionary to store named loggers
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._configure_base_logging()
            self._configure_third_party_loggers()
            self._initialized = True

    def _configure_base_logging(self):
        """Configure base logging settings"""
        # Get environment settings
        env_setting = env.get('LOGGING_ENV', 'development').lower()
        logging_to_file = env.get('LoggingtoFile', 'false').lower() == 'true'

        # Configure root logger
        root_logger = logging.getLogger()
        if not root_logger.handlers:  # Only configure if not already configured
            # Create console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

            # Add file handler if enabled
            if logging_to_file:
                try:
                    log_dir = 'logs'
                    os.makedirs(log_dir, exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
                    log_file = os.path.join(log_dir, f'app_{timestamp}.log')
                    
                    file_handler = RotatingFileHandler(
                        log_file,
                        maxBytes=30485760,  # 30MB
                        backupCount=5
                    )
                    file_handler.setFormatter(formatter)
                    root_logger.addHandler(file_handler)
                except Exception as e:
                    console_handler.setLevel(logging.WARNING)
                    root_logger.warning(f"Failed to initialize file logging: {e}")

            # Set level based on environment
            log_level = logging.WARNING if env_setting == 'production' else logging.INFO
            root_logger.setLevel(log_level)

    def _configure_third_party_loggers(self):
        """Configure logging levels for third-party libraries"""
        env_setting = env.get('LOGGING_ENV', 'development').lower()
        
        # List of third-party loggers to configure
        third_party_loggers = [
            'httpx',
            'urllib3',
            'requests',
            'pinecone',
            'boto3',
            'botocore',
            'openai',
            'auth0',
            'werkzeug',
            'flask'
        ]

        # Set appropriate level based on environment
        level = logging.WARNING if env_setting == 'production' else logging.INFO

        # Configure each third-party logger
        for logger_name in third_party_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            # Prevent propagation to root logger to avoid duplicate logs
            logger.propagate = False

            # Ensure the logger has at least one handler
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance with the singleton configuration"""
        if cls._instance is None:
            cls()

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger

        return cls._loggers[name] 