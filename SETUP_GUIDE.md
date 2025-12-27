# Claude Code OpenAI Wrapper - Docker Setup Guide

This guide will help you set up the Claude Code OpenAI Wrapper using Docker, allowing you to use your **Claude Pro/Max subscription** as an OpenAI-compatible API.

## Why Use This?

| Approach | Cost |
|----------|------|
| Claude API (pay-per-token) | ~$3-15 per 1M tokens |
| **Claude Code CLI (subscription)** | **$20/mo flat rate** |

This wrapper exposes your Claude Code CLI subscription as a REST API, so you can integrate it with tools like **n8n**, **LangChain**, or any OpenAI-compatible client.

---

## Prerequisites

- **Docker Desktop** installed and running
- **Claude Pro or Max subscription** (the wrapper uses your subscription, not API tokens)

---

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/projectfr33d0m2025/claude-code-openai-wrapper.git
cd claude-code-openapi-wrapper
```

### Step 2: Build the Docker Image

```bash
docker-compose build
```

> **Note:** First build takes 5-10 minutes (installs Node.js, Claude CLI, Python dependencies).

### Step 3: Start the Container

```bash
docker-compose up -d
```

### Step 4: Authenticate Claude (Required - First Time Only)

Since Claude Code stores authentication in the system keychain (which Docker can't access), you need to authenticate inside the container:

```bash
docker exec -it claude-code-wrapper claude auth login
```

This will:
1. Display a URL and code
2. Open your browser for OAuth authentication
3. Store the tokens in the mounted `~/.claude` directory

> **Important:** The tokens persist in `~/.claude` on your host, so you only need to do this once (unless tokens expire).

### Step 5: Verify Everything Works

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check authentication status
curl http://localhost:8000/v1/auth/status

# List available models
curl http://localhost:8000/v1/models
```

### Step 6: Test a Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [{"role": "user", "content": "Hello! What is 2+2?"}]
  }'
```

---

## Using with n8n

### Option A: HTTP Request Node

In your n8n workflow, add an **HTTP Request** node:

| Setting | Value |
|---------|-------|
| Method | POST |
| URL | `http://host.docker.internal:8000/v1/chat/completions` |
| Headers | `Content-Type: application/json` |

**Body (JSON):**
```json
{
  "model": "claude-sonnet-4-5-20250929",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "{{ $json.prompt }}"}
  ]
}
```

### Option B: OpenAI Node (with custom base URL)

1. Create an OpenAI credential with any dummy API key
2. In the node settings, set **Base URL** to: `http://host.docker.internal:8000/v1`
3. Select your preferred Claude model

### URL Options by Setup

| n8n Setup | URL to Use |
|-----------|------------|
| n8n in same Docker network | `http://claude-code-wrapper:8000/v1/chat/completions` |
| n8n in separate container | `http://host.docker.internal:8000/v1/chat/completions` |
| n8n on host machine | `http://localhost:8000/v1/chat/completions` |

---

## Available Models

| Model | Description |
|-------|-------------|
| `claude-sonnet-4-5-20250929` | **Recommended** - Best balance of speed and capability |
| `claude-haiku-4-5-20251001` | Fastest, most economical |
| `claude-opus-4-1-20250805` | Most capable, slower |

---

## Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop the service
docker-compose down

# Restart the service
docker-compose restart

# Rebuild after changes
docker-compose build --no-cache
docker-compose up -d

# Check container status
docker-compose ps

# Re-authenticate (if tokens expire)
docker exec -it claude-code-wrapper claude auth login

# Test Claude CLI directly in container
docker exec -it claude-code-wrapper claude --print "Hello"
```

---

## Troubleshooting

### "Invalid API key" or Authentication Errors

```bash
# Re-authenticate inside the container
docker exec -it claude-code-wrapper claude auth login
```

### Connection Refused

- Wait 30-60 seconds after starting for the service to initialize
- Check logs: `docker-compose logs -f`

### Timeout Errors

Increase `MAX_TIMEOUT` in `docker-compose.yml`:
```yaml
environment:
  - MAX_TIMEOUT=900000  # 15 minutes
```

### Container Won't Start

```bash
# Check what's wrong
docker-compose logs

# Try rebuilding
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## Configuration Options

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - PORT=8000                              # API port
  - MAX_TIMEOUT=600000                     # Request timeout (ms)
  - DEFAULT_MODEL=claude-sonnet-4-5-20250929  # Default model
  - RATE_LIMIT_ENABLED=true                # Enable rate limiting
  - CORS_ORIGINS=["*"]                     # Allowed CORS origins
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `GET /v1/auth/status` | Check authentication status |
| `POST /v1/chat/completions` | Chat completions (OpenAI-compatible) |
| `GET /v1/sessions` | List active sessions |
| `DELETE /v1/sessions/{id}` | Delete a session |

---

## Cost

This setup uses your **Claude Pro/Max subscription** ($20/month flat rate).

**No API tokens are consumed** - you're using the CLI subscription quota, which is significantly more generous than API pricing.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

Based on [RichardAtCT/claude-code-openai-wrapper](https://github.com/RichardAtCT/claude-code-openai-wrapper).
