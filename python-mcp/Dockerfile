# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install build tools minimally (for wheels if needed)
RUN pip install --no-cache-dir --upgrade pip

# Copy project metadata first to leverage Docker layer caching
COPY pyproject.toml ./
COPY README.md ./

# Install project dependencies from pyproject (no source yet)
RUN pip install --no-cache-dir .

# Copy source last
COPY . .

# Optional: non-root
RUN useradd -m runner && chown -R runner:runner /app
USER runner

# Serve MCP over HTTP using FastMCP transport
# Azure binds $PORT; default to 8000 locally
ENV PORT=8000
CMD ["sh", "-c", "fastmcp run papr_memory_mcp.core:init_mcp --transport http --host 0.0.0.0 --port ${PORT}"]


