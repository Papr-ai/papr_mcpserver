"""
Papr Memory MCP Server

A FastAPI-based MCP (Model Context Protocol) server implementation for integrating 
with Papr's memory services (https://papr.ai).

This package provides tools for memory management through the MCP protocol,
including adding, retrieving, updating, and searching memories.
"""

__version__ = "0.1.0"
__author__ = "Papr Team"
__email__ = "support@papr.ai"

from .paprmcp import main, CustomFastMCP, init_mcp

__all__ = ["main", "CustomFastMCP", "init_mcp"]
