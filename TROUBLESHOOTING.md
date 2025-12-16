# Troubleshooting "Connection Reset by Peer" Error

## Problem
Getting "connection reset by peer" error when trying to connect to the API.

## Root Cause
The server process is crashing (segfault) during startup, typically due to:
1. Model initialization issues during lifespan startup
2. CUDA/GPU library initialization failures
3. Library incompatibilities causing crashes

## Solution 1: Disable Startup Model Initialization (Recommended)

The lifespan function in `api.py` now skips model initialization on startup. Models are initialized lazily on first request.

If you still have issues, verify the change was applied:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the layout detection model on startup."""
    # Startup - skip model initialization on startup to avoid crashes
    print("FastAPI application starting...")
    print("Models will be initialized on first request.")
    
    yield
```

## Solution 2: Check Container Logs

```bash
# View recent logs
docker-compose logs --tail=100 api

# Follow logs in real-time
docker-compose logs -f api

# Check for errors
docker-compose logs api | grep -i error
```

## Solution 3: Test Without Reload

The `--reload` flag can sometimes mask errors. Test without it:

```yaml
# In docker-compose.yml, temporarily use:
command: ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
```

## Solution 4: Verify API Can Start

Test if the API module can be imported:

```bash
docker-compose exec api python -c "import sys; sys.path.insert(0, '/app'); import api; print('OK')"
```

If this crashes with exit code 139, there's a segfault during import.

## Solution 5: Check GPU Access (for GPU version)

If using GPU docker-compose:

```bash
# Verify GPU is accessible
docker-compose exec api nvidia-smi

# If GPU not available, models might crash. Consider:
# 1. Using CPU version instead
# 2. Ensuring NVIDIA Container Toolkit is installed
# 3. Running with: docker-compose -f docker-compose.yml (CPU version)
```

## Solution 6: Increase Container Resources

If crashes are due to memory issues:

```yaml
# In docker-compose.yml, add:
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

## Solution 7: Use Production Dockerfile

The dev dockerfile mounts volumes which can cause issues. Try production version:

```yaml
# In docker-compose.yml, change:
dockerfile: Dockerfile  # instead of Dockerfile.dev
# And remove volume mounts temporarily
```

## Quick Diagnostic Commands

```bash
# Check container status
docker-compose ps

# Check if port is listening
docker-compose exec api netstat -tlnp | grep 8000

# Test from inside container
docker-compose exec api curl http://localhost:8000/

# Check container resources
docker stats pdf-api-1
```

## Expected Successful Startup Logs

You should see:
```
INFO:     Started server process [X]
INFO:     Waiting for application startup.
FastAPI application starting...
Models will be initialized on first request.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If you see "INFO:     Started reloader process" but no "Started server process", the server is crashing.

