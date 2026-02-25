#!/bin/bash
#
# KBV2 Server Start Script
# One-stop script to start the Portable Knowledge Base server
#
# Usage: ./start.sh [--port PORT] [--reload]
#

set -e

# Configuration
DEFAULT_PORT=8088
PORT=${DEFAULT_PORT}
RELOAD=""
HOST="0.0.0.0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--port PORT] [--reload]"
            echo ""
            echo "Options:"
            echo "  --port PORT   Port to run server on (default: $DEFAULT_PORT)"
            echo "  --reload      Enable auto-reload for development"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Start server on port $DEFAULT_PORT"
            echo "  $0 --port 9000        # Start server on port 9000"
            echo "  $0 --reload           # Start with auto-reload for development"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "KBV2 - Portable Knowledge Base Server"
echo "========================================"
echo ""

# Function to kill process on port
kill_port_process() {
    local port=$1
    local pid=$(lsof -t -i :$port 2>/dev/null)

    if [ -n "$pid" ]; then
        echo "⚠️  Found process on port $port (PID: $pid)"
        echo "   Killing process..."
        kill -9 $pid 2>/dev/null || sudo kill -9 $pid 2>/dev/null
        sleep 1

        # Verify port is free
        if lsof -i :$port > /dev/null 2>&1; then
            echo "❌ Failed to free port $port"
            exit 1
        fi
        echo "✅ Port $port freed"
    fi
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 1  # Port is in use
    fi
    return 0  # Port is free
}

# Function to check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."

    # Check uv
    if ! command -v uv &> /dev/null; then
        echo "❌ uv is not installed"
        echo "   Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "   ✅ uv installed"

    # Check Ollama for embeddings
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ✅ Ollama running"
    else
        echo "   ⚠️  Ollama not running (embeddings will fail)"
        echo "      Start with: ollama serve"
    fi

    # Check LLM gateway
    if curl -s http://localhost:8087/v1/models > /dev/null 2>&1; then
        echo "   ✅ LLM gateway running"
    else
        echo "   ⚠️  LLM gateway not running at http://localhost:8087"
    fi

    echo ""
}

# Check and create data directory
ensure_data_dir() {
    if [ ! -d "data" ]; then
        echo "📁 Creating data directory..."
        mkdir -p data
    fi
}

# Main script
echo "Checking port $PORT..."
if ! check_port $PORT; then
    kill_port_process $PORT
fi
echo "✅ Port $PORT available"
echo ""

check_prerequisites
ensure_data_dir

# Start server
echo "🚀 Starting KBV2 server on port $PORT..."
echo ""

if [ -n "$RELOAD" ]; then
    echo "   Mode: Development (auto-reload enabled)"
else
    echo "   Mode: Production"
fi
echo "   Host: $HOST"
echo "   Port: $PORT"
echo ""
echo "Endpoints:"
echo "   Health:  http://localhost:$PORT/health"
echo "   Stats:   http://localhost:$PORT/stats"
echo "   Docs:    http://localhost:$PORT/redoc"
echo ""
echo "Press Ctrl+C to stop"
echo "----------------------------------------"
echo ""

# Run uvicorn
exec uv run uvicorn knowledge_base.main:app --host $HOST --port $PORT $RELOAD
