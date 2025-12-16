# Use Python 3.11 slim as base image
# For GPU support (required for paddlepaddle-gpu), use:
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official container image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files needed for installation
COPY pyproject.toml ./
COPY pdf_extract_kit/ ./pdf_extract_kit/

# Install dependencies and the package using uv
# uv pip install with . installs the package and all dependencies from pyproject.toml
RUN uv pip install --system --no-cache .

# Copy configuration files
COPY configs/ ./configs/

# Note: api.py and models/ are copied here for standalone/production use
# When using docker-compose with volumes, these will be overridden by the mounted volumes
# If you only use docker-compose for dev, you can remove the COPY commands below
COPY models/ ./models/
COPY api.py ./

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
