#!/usr/bin/env python3
"""
Setup script for papr-memory-mcp package.
This is a fallback for tools that don't support pyproject.toml yet.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="papr-memory-mcp",
    version="0.1.0",
    author="Papr Team",
    author_email="support@papr.ai",
    description="Papr Memory API integration for MCP (Model Context Protocol) servers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Papr-ai/papr_mcpserver",
    project_urls={
        "Homepage": "https://github.com/Papr-ai/papr_mcpserver",
        "Documentation": "https://github.com/Papr-ai/papr_mcpserver/tree/main/python-mcp",
        "Repository": "https://github.com/Papr-ai/papr_mcpserver",
        "Issues": "https://github.com/Papr-ai/papr_mcpserver/issues",
        "Changelog": "https://github.com/Papr-ai/papr_mcpserver/blob/main/python-mcp/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "debugpy",
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "flask-socketio>=5.0.0",
            "python-socketio>=5.0.0",
            "eventlet>=0.30.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
        ],
        "ml": [
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "langchain-community>=0.0.10",
            "ollama>=0.1.6",
            "sentence-transformers>=2.2.2",
            "transformers>=4.36.0",
            "langchain>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "papr-mcp=papr_memory_mcp.paprmcp:main",
            "papr-memory-mcp=papr_memory_mcp.paprmcp:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
