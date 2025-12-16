# Testing API Connectivity

## Quick Test Methods

### 1. Using curl (Quickest)

```bash
# Test root endpoint (GET /)
curl http://localhost:8000/

# Test with pretty JSON output
curl http://localhost:8000/ | python -m json.tool

# Check if API is responding (just HTTP status)
curl -I http://localhost:8000/
```

### 2. Using Python script

```bash
# Basic test (uses default http://localhost:8000)
python test_api.py

# Test custom URL
python test_api.py http://localhost:8000
```

### 3. Using httpie (if installed)

```bash
# Install httpie: pip install httpie
http GET http://localhost:8000/
```

### 4. Using browser

Open in your browser:
- **API Root**: http://localhost:8000/
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### 5. Using wget

```bash
wget -qO- http://localhost:8000/
```

## Testing from Inside Container

```bash
# Connect to running container
docker-compose exec api bash

# Inside container, test with curl
curl http://localhost:8000/
```

## Testing with Docker

```bash
# Test if container is running
docker-compose ps

# Check logs
docker-compose logs -f api

# Test connectivity from host
docker run --rm --network pdf_default curl http://api:8000/
```

## Expected Response

Successful root endpoint response:
```json
{
  "Hello": "World",
  "endpoints": [
    "/detect-layout",
    "/ocr",
    "/formula-recognition",
    "/formula-detection"
  ]
}
```

## Troubleshooting

1. **Connection refused**: Container not running
   ```bash
   docker-compose up -d
   ```

2. **Port already in use**: Another service on port 8000
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   # Or change port in docker-compose.yml
   ```

3. **Container keeps restarting**: Check logs
   ```bash
   docker-compose logs api
   ```

