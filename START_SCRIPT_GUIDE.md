# Start Script Guide

## Overview

The `start.sh` script provides a one-command solution to launch the entire AI Assessment Platform with automatic browser UI opening.

## Usage

```bash
bash start.sh
```

Or make it executable and run directly:

```bash
chmod +x start.sh
./start.sh
```

## What It Does

### 1. Environment Cleanup
```
🧹 Cleaning environment variables...
   ✓ QDRANT_HOST unset (will use 'qdrant' container name)
   ✓ QDRANT_SEED_PATH unset (will use container path)
```

Prevents shell environment variables from conflicting with Docker container configurations.

### 2. Container Startup
```
🐳 Starting Docker containers...
✅ Containers started successfully!
```

Launches all services:
- Backend API (FastAPI)
- Frontend UI (Streamlit)
- Qdrant vector database
- Inference service (ONNX)

### 3. Health Checks
```
🔍 Checking backend API...
   ✓ Backend API is ready
🔍 Checking Qdrant...
   ✓ Qdrant is ready
```

Waits for services to be fully operational before proceeding.

### 4. Browser Auto-Open
```
🌐 Opening frontend in browser...
🎉 Ready to use! The UI should open automatically.
```

Automatically opens http://localhost:8501 in your default browser.

## Platform Support

### macOS
Uses the `open` command:
```bash
open http://localhost:8501
```

### Linux
Tries multiple commands in order:
```bash
xdg-open http://localhost:8501      # Most Linux distros
gnome-open http://localhost:8501    # GNOME desktop
```

### Windows
Uses the `start` command (Git Bash/Cygwin):
```bash
start http://localhost:8501
```

### Fallback
If auto-open fails, shows manual URL:
```
ℹ️  Please manually open: http://localhost:8501
```

## Output Example

```
==========================================
🚀 Starting AI Assessment Platform
==========================================

🧹 Cleaning environment variables...
   ✓ QDRANT_HOST unset (will use 'qdrant' container name)
   ✓ QDRANT_SEED_PATH unset (will use container path)

🐳 Starting Docker containers...

✅ Containers started successfully!

⏳ Waiting for services to start...
🔍 Checking backend API...
   ✓ Backend API is ready
🔍 Checking Qdrant...
   ✓ Qdrant is ready

==========================================
✅ AI Assessment Platform Started!
==========================================

📊 Available Services:
   • Frontend:        http://localhost:8501
   • Backend API:     http://localhost:8888
   • API Docs:        http://localhost:8888/docs
   • Qdrant:          http://localhost:6333/dashboard

📝 Useful Commands:
   • View logs:       docker-compose logs -f
   • Stop services:   docker-compose down
   • Restart:         docker-compose restart
   • Check status:    docker-compose ps

🌐 Opening frontend in browser...

🎉 Ready to use! The UI should open automatically.
   If not, visit: http://localhost:8501

==========================================
```

## Useful Commands After Startup

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f qdrant
```

### Check Status
```bash
docker-compose ps
```

### Stop Services
```bash
docker-compose down
```

### Restart a Service
```bash
docker-compose restart backend
docker-compose restart frontend
```

## Troubleshooting

### Browser Doesn't Open Automatically

**Manual access:**
```bash
# Open in browser
open http://localhost:8501        # macOS
xdg-open http://localhost:8501    # Linux
start http://localhost:8501       # Windows
```

Or just navigate to: http://localhost:8501

### Health Check Fails

**Backend not responding:**
```bash
# Check backend logs
docker-compose logs backend

# Restart backend
docker-compose restart backend
```

**Qdrant not responding:**
```bash
# Check Qdrant logs
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant
```

### Port Already in Use

**Check what's using the ports:**
```bash
# Check port 8501 (Frontend)
lsof -i :8501

# Check port 8888 (Backend)
lsof -i :8888

# Check port 6333 (Qdrant)
lsof -i :6333
```

**Kill the process:**
```bash
# Find and kill process on port 8501
lsof -ti :8501 | xargs kill -9
```

### Containers Won't Start

**Check Docker daemon:**
```bash
docker info
```

**Check docker-compose configuration:**
```bash
docker-compose config
```

**Rebuild containers:**
```bash
docker-compose down
docker-compose build --no-cache
bash start.sh
```

## Advanced Usage

### Pass Arguments to docker-compose

```bash
# Rebuild containers before starting
bash start.sh --build

# Start in foreground (see logs)
bash start.sh --no-detach

# Start specific services
bash start.sh backend frontend
```

### Environment Variable Override

```bash
# Use different environment file
ENV_FILE=.env.production bash start.sh

# Override specific variables
QDRANT_PORT=6334 bash start.sh
```

## Integration with Development Workflow

### Quick Restart During Development

```bash
# Stop containers
docker-compose down

# Rebuild and restart
bash start.sh --build
```

### View Logs While Working

```bash
# Terminal 1: Start services
bash start.sh

# Terminal 2: Watch logs
docker-compose logs -f backend
```

### Debug Mode

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Start with verbose output
bash start.sh
```

## Script Internals

### Health Check Logic

The script waits up to 20 seconds for the backend API to respond:

```bash
for i in {1..10}; do
    if curl -s http://localhost:8888/health > /dev/null 2>&1; then
        echo "   ✓ Backend API is ready"
        break
    fi
    sleep 2
done
```

### OS Detection

Uses Bash's `$OSTYPE` variable:

```bash
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8501
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:8501
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start http://localhost:8501
fi
```

## Customization

### Change Startup Delay

Edit the sleep duration in `start.sh`:

```bash
# Default: wait 3 seconds
sleep 3

# Custom: wait 5 seconds
sleep 5
```

### Disable Auto-Open

Comment out the browser opening section:

```bash
# # Open browser to frontend UI
# echo "🌐 Opening frontend in browser..."
# sleep 2
#
# if [[ "$OSTYPE" == "darwin"* ]]; then
#     open http://localhost:8501
# fi
```

### Add Custom Health Checks

Add checks for additional services:

```bash
# Check inference service
echo "🔍 Checking inference service..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✓ Inference service is ready"
fi
```

## Best Practices

1. **Always use start.sh instead of `docker-compose up` directly**
   - Ensures environment variables are clean
   - Provides health checks
   - Auto-opens UI

2. **Check the output for warnings**
   - Yellow warnings indicate services that didn't start properly
   - Use suggested commands to debug

3. **Keep the script updated**
   - If you add new services to docker-compose.yml
   - Update health checks in start.sh

4. **Use in CI/CD**
   - Can run start.sh in CI environments
   - Disable browser opening with environment variable

## Summary

**One command to start everything:**
```bash
bash start.sh
```

**What you get:**
- ✅ All containers started
- ✅ Health checks passed
- ✅ Browser opens automatically
- ✅ Clear status messages
- ✅ Helpful command reference

**Time saved:** ~30 seconds per startup (no manual URL navigation, no service checking)

---

**Need help?** Check the troubleshooting section or view logs with:
```bash
docker-compose logs -f
```
