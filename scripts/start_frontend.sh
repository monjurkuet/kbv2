#!/bin/bash
# KBV2 Frontend Startup Script
# Usage: ./scripts/start_frontend.sh [options]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
cd "$PROJECT_ROOT"

log_info() {
    echo -e "${BLUE}[FRONTEND]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[FRONTEND]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[FRONTEND]${NC} $1"
}

log_error() {
    echo -e "${RED}[FRONTEND]${NC} $1"
    exit 1
}

check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v bun &>/dev/null; then
        log_error "bun not found. Run setup_kbv2.sh first"
        exit 1
    fi
    
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        log_error "node_modules not found. Run setup_kbv2.sh first"
        exit 1
    fi
}

wait_for_api() {
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for backend API to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8765/health >/dev/null; then
            log_success "Backend API is ready"
            return 0
        fi
        
        log_info "Waiting for API... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_warning "Backend API not ready after $max_attempts attempts"
    return 1
}

generate_api_client() {
    log_info "Generating API client..."
    
    cd "$PROJECT_ROOT"
    
    # Re-generate OpenAPI schema if backend is available
    if curl -s http://localhost:8765/openapi.json >/dev/null 2>&1; then
        log_info "Fetching OpenAPI schema from running server..."
        curl -s http://localhost:8765/openapi.json > frontend/openapi-schema.json
    elif [ -f scripts/generate_openapi.py ]; then
        log_info "Generating OpenAPI schema from backend code..."
        uv run python scripts/generate_openapi.py 2>/dev/null || log_warning "Schema generation may have issues"
    fi
    
    cd "$FRONTEND_DIR"
    
    if [ -f openapi-schema.json ]; then
        bun run api:generate
        log_success "API client generated"
    else
        log_warning "OpenAPI schema not found. API client may be outdated."
    fi
}

start_frontend() {
    cd "$FRONTEND_DIR"
    
    log_info "Starting Vite dev server on port 3000..."
    log_info "Access URL: http://localhost:3000"
    
    bun run dev
}

main() {
    check_requirements
    
    # Only wait for API if it's likely running
    if curl -s http://localhost:8765/health >/dev/null 2>&1; then
        wait_for_api
    else
        log_warning "Backend API not running. Make sure to start it with start_backend.sh"
    fi
    
    generate_api_client
    start_frontend
}

main
