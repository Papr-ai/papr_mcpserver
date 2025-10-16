# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Copy EVERYTHING first so pip can build your package
# (your previous layer installed "." before sources were present)
COPY . .

# Install your package (and fastmcp via pyproject deps)
# If fastmcp isn't in your pyproject, add: && pip install fastmcp
RUN pip install --no-cache-dir .

# Optional: non-root
RUN useradd -m runner && chown -R runner:runner /app
USER runner

# Azure may set PORT or WEBSITE_PORT; default 8080
ENV PORT=${PORT}
ENV WEBSITE_PORT=${WEBSITE_PORT}

# Serve MCP over HTTP
# IMPORTANT: pass a FILE path (not a module path) to fastmcp run
CMD sh -c 'fastmcp run papr_memory_mcp/core.py:init_mcp --transport http --host 0.0.0.0 --port "${PORT:-${WEBSITE_PORT:-8080}}"'
