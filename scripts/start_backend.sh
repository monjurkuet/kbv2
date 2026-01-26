#!/bin/bash
# KBV2 Backend Startup Script
# Usage: ./scripts/start_backend.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$PROJECT_ROOT"

log_info() {
    echo -e "${BLUE}[BACKEND]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[BACKEND]${NC} $1"
}

log_error() {
    echo -e "${RED}[BACKEND]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v uv &>/dev/null; then
        log_error "uv not found. Run setup_kbv2.sh first"
        exit 1
    fi
    
    if [ ! -f .env ]; then
        log_error ".env file not found"
        exit 1
    fi
    
    # Re-generate OpenAPI schema to ensure it's up to date
    log_info "Generating OpenAPI schema..."
    uv run python scripts/generate_openapi.py
    
    log_success "Requirements check passed"
}

start_server() {
    log_info "Starting FastAPI server on port 8765..."
    log_info "API Docs: http://localhost:8765/docs"
    log_info "Health: http://localhost:8765/health"
    
    # Set PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    # Start the server
    uv run knowledge-base
}

main() {
    check_requirements
    start_server
}

main
