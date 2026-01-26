#!/bin/bash
# KBV2 Development Script - Runs backend and frontend together
# Usage: ./scripts/dev.sh

set -euo pipefail  # Strict error handling

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

# Configuration
BACKEND_PORT=8765
FRONTEND_PORT=3000
MAX_HEALTH_CHECKS=30
HEALTH_CHECK_INTERVAL=2
MAX_PORT_WAIT=15  # Increased for more robust waiting

# Global PIDs
BACKEND_PID=""
FRONTEND_PID=""

# Cleanup on exit
cleanup() {
    local exit_code=$?
    echo -e "\n${YELLOW}[DEV]${NC} ${BOLD}Initiating cleanup...${NC}"
    
    # Kill backend processes
    if [ -n "${BACKEND_PID:-}" ]; then
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            log_info "Terminating backend (PID: $BACKEND_PID)..."
            kill -9 "$BACKEND_PID" 2>/dev/null || true
            sleep 1
        fi
        pkill -9 -f "uv run knowledge-base" 2>/dev/null || true
        pkill -9 -f "uvicorn" 2>/dev/null || true
        log_success "Backend processes terminated"
    fi
    
    # Kill frontend processes  
    if [ -n "${FRONTEND_PID:-}" ]; then
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            log_info "Terminating frontend (PID: $FRONTEND_PID)..."
            kill -9 "$FRONTEND_PID" 2>/dev/null || true
            sleep 1
        fi
        pkill -9 -f "vite" 2>/dev/null || true
        pkill -9 -f "bun.*dev" 2>/dev/null || true
        log_success "Frontend processes terminated"
    fi
    
    # Wait for ports to be released with timeout (script continues even if ports aren't released)
    log_info "Waiting for ports to be released..."
    local port_wait_start=$(date +%s)
    
    wait_for_port_release $BACKEND_PORT "Backend"
    wait_for_port_release $FRONTEND_PORT "Frontend"
    
    local port_wait_time=$(($(date +%s) - port_wait_start))
    log_success "Cleanup completed in ${port_wait_time}s"
    
    # Exit with original code or 0 if called normally
    exit $exit_code
}

trap cleanup SIGINT SIGTERM EXIT ERR

echo() {
    builtin echo -e "$(date '+%H:%M:%S') $*"
}

log_info() {
    echo "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo "${RED}[ERROR]${NC} $1"
}

# Check if a port is in use
is_port_in_use() {
    local port=$1
    lsof -i :"$port" >/dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port"
}

# Wait for port to be released with progress indication
wait_for_port_release() {
    local port=$1
    local name=$2
    local attempts=0
    
    while is_port_in_use "$port" && [ $attempts -lt $MAX_PORT_WAIT ]; do
        if [ $((attempts % 5)) -eq 0 ]; then
            log_info "Waiting for $name port $port to be released... (attempt $((attempts + 1))/$MAX_PORT_WAIT)"
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    
    if is_port_in_use "$port"; then
        log_warning "Port $port may still be in use after $attempts attempts, continuing anyway..."
        return 1
    else
        log_success "Port $port ($name) is now free"
        return 0
    fi
}

# Wait for service to be responsive
wait_for_service() {
    local port=$1
    local name=$2
    local health_check=${3:-}  # Optional health endpoint
    local attempts=0
    
    while [ $attempts -lt $MAX_HEALTH_CHECKS ]; do
        if [ -n "$health_check" ]; then
            # Health endpoint check (more reliable)
            if health_response=$(curl -s "http://localhost:${port}/${health_check}" 2>/dev/null); then
                if echo "$health_response" | grep -qE '"?(status|ok)"?\s*:\s*(true|"ok"|"healthy")'; then
                    return 0
                fi
            fi
        else
            # Basic port check
            if curl -s "http://localhost:$port" >/dev/null 2>&1 || nc -z localhost "$port" 2>/dev/null; then
                return 0
            fi
        fi
        
        # Progress indicator every 5 attempts
        if [ $((attempts % 5)) -eq 0 ]; then
            log_info "Waiting for $name... (attempt $((attempts + 1))/$MAX_HEALTH_CHECKS)"
        fi
        
        sleep $HEALTH_CHECK_INTERVAL
        attempts=$((attempts + 1))
    done
    
    return 1
}

# Print startup banner
print_banner() {
    echo -e "${GREEN}${BOLD}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                    KBV2 Development Mode                     ║
║                                                              ║
║  Starting both backend and frontend                          ║
║  Press Ctrl+C to stop both services                          ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo "Backend:  ${BOLD}http://localhost:${BACKEND_PORT}${NC}"
    echo "Frontend: ${BOLD}http://localhost:${FRONTEND_PORT}${NC}"
    echo "API Docs: ${BOLD}http://localhost:${BACKEND_PORT}/docs${NC}"
    echo ""
    log_info "Log timestamps: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# Verify all requirements are met
check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_reqs=()
    
    if ! command -v uv &>/dev/null; then
        missing_reqs+=("uv")
    fi
    
    if ! command -v bun &>/dev/null; then
        missing_reqs+=("bun")
    fi
    
    if ! command -v curl &>/dev/null; then
        missing_reqs+=("curl")
    fi
    
    if ! command -v lsof &>/dev/null && ! command -v netstat &>/dev/null && ! command -v ss &>/dev/null; then
        missing_reqs+=("lsof or netstat or ss")
    fi
    
    if [ ${#missing_reqs[@]} -ne 0 ]; then
        log_error "Missing required commands: ${missing_reqs[*]}"
        log_error "Please run setup_kbv2.sh first"
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_error ".env file not found in $PROJECT_ROOT"
        exit 1
    fi
    
    if [ ! -d "$PROJECT_ROOT/frontend/node_modules" ]; then
        log_error "Frontend dependencies not installed"
        log_warning "Run: cd frontend && bun install"
        exit 1
    fi
    
    log_success "All requirements met"
}

# Clear ports before starting
clear_ports() {
    log_info "Checking for processes using required ports..."
    
    local ports_to_clear=()
    is_port_in_use $BACKEND_PORT && ports_to_clear+=("$BACKEND_PORT")
    is_port_in_use $FRONTEND_PORT && ports_to_clear+=("$FRONTEND_PORT")
    
    if [ ${#ports_to_clear[@]} -gt 0 ]; then
        log_warning "Ports ${ports_to_clear[*]} are in use, terminating existing processes..."
        
        for port in "${ports_to_clear[@]}"; do
            log_info "Clearing port $port..."
            local pids=$(sudo lsof -i :$port -t 2>/dev/null)
            
            if [ -n "$pids" ]; then
                for pid in $pids; do
                    log_info "Killing process $pid on port $port..."
                    sudo kill -9 $pid 2>/dev/null || true
                done
            fi
        done
        
        # Wait for ports to be released
        wait_for_port_release $BACKEND_PORT "Backend"
        wait_for_port_release $FRONTEND_PORT "Frontend"
    else
        log_success "All required ports are available"
    fi
}

# Start backend service
start_backend() {
    log_info "Starting backend service..."
    
    cd "$PROJECT_ROOT"
    
    # Generate OpenAPI schema
    log_info "Generating OpenAPI schema..."
    if ! uv run python scripts/generate_openapi.py 2>&1 | tee /tmp/backend_schema.log; then
        log_error "Failed to generate OpenAPI schema"
        log_error "Check /tmp/backend_schema.log for details"
        exit 1
    fi
    log_success "OpenAPI schema generated"
    
    export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
    
    # Verify port is still free
    if is_port_in_use $BACKEND_PORT; then
        log_error "Port $BACKEND_PORT became occupied, aborting"
        exit 1
    fi
    
    # Start backend
    log_info "Launching backend (uv run knowledge-base)..."
    exec 3>&1 4>&2  # Save stdout/stderr
    uv run knowledge-base > /tmp/backend.log 2>&1 &
    BACKEND_PID=$!
    exec 1>&3 2>&4  # Restore stdout/stderr
    
    log_info "Backend started with PID: $BACKEND_PID"
    
    # Multi-attempt health check with progress
    log_info "Waiting for backend health endpoint..."
    if ! wait_for_service $BACKEND_PORT "backend" "health"; then
        log_error "Backend failed to become healthy after $MAX_HEALTH_CHECKS attempts"
        log_error "Last backend log lines:"
        tail -20 /tmp/backend.log 2>/dev/null | sed 's/^/[BACKEND] /' || true
        log_error "Health check output:"
        curl -v "http://localhost:${BACKEND_PORT}/health" 2>&1 || echo "No response"
        exit 1
    fi
    
    log_success "Backend is healthy and ready"
}

# Start frontend service
start_frontend() {
    log_info "Starting frontend service..."
    
    cd "$PROJECT_ROOT/frontend"
    
    # Generate API client
    log_info "Generating API client..."
    if ! bun run api:generate 2>&1 | tee /tmp/frontend_api_gen.log; then
        log_warning "API client generation had issues (check /tmp/frontend_api_gen.log)"
    else
        log_success "API client generated"
    fi
    
    # Verify port is free
    if is_port_in_use $FRONTEND_PORT; then
        log_error "Port $FRONTEND_PORT became occupied, aborting"
        exit 1
    fi
    
    # Start frontend  
    log_info "Launching frontend (bun run dev)..."
    exec 3>&1 4>&2  # Save stdout/stderr
    bun run dev > /tmp/frontend.log 2>&1 &
    FRONTEND_PID=$!
    exec 1>&3 2>&4  # Restore stdout/stderr
    
    log_info "Frontend started with PID: $FRONTEND_PID"
    
    # Wait for frontend to be responsive
    log_info "Waiting for frontend to be responsive..."
    if ! wait_for_service $FRONTEND_PORT "frontend"; then
        log_error "Frontend failed to start after $MAX_HEALTH_CHECKS attempts"
        log_error "Last frontend log lines:"
        tail -20 /tmp/frontend.log 2>/dev/null | sed 's/^/[FRONTEND] /' || true
        exit 1
    fi
    
    log_success "Frontend is ready"
}

# Monitor running processes
monitor_processes() {
    log_info "${BOLD}Both services are running successfully!${NC}"
    log_info "Press Ctrl+C to stop all services"
    echo ""
    
    # Show live logs in a more readable format
    log_info "Showing service logs (timestamped):"
    echo ""
    
    tail -f /tmp/backend.log /tmp/frontend.log 2>/dev/null | while read -r line; do
        echo "$(date '+%H:%M:%S') $line"
    done &
    local tail_pid=$!
    
    # Monitor process health
    while true; do
        # Check backend
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            log_error "Backend process (PID: $BACKEND_PID) has died unexpectedly"
            break
        fi
        
        # Check frontend
        if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
            log_error "Frontend process (PID: $FRONTEND_PID) has died unexpectedly"
            break
        fi
        
        # Periodic health check
        if [ $(($(date +%s) % 30)) -eq 0 ]; then
            if ! curl -s "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
                log_warning "Backend health check failed at $(date '+%H:%M:%S')"
            fi
        fi
        
        sleep 2
    done
    
    # Kill tail when exiting
    kill $tail_pid 2>/dev/null || true
    
    log_error "Service monitoring detected a failure"
    exit 1
}

main() {
    # Save main script PID
    MAIN_PID=$$
    
    print_banner
    
    # Pre-flight checks
    check_requirements
    clear_ports
    
    # Start services sequentially
    start_backend
    
    # Small buffer after backend is healthy
    log_info "Backend buffer pause..."
    sleep 2
    
    start_frontend
    
    # Final verification
    log_info "Verifying both services are responsive..."
    wait_for_service $BACKEND_PORT "backend" "health"
    wait_for_service $FRONTEND_PORT "frontend"
    
    # Show final status
    echo ""
    echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}${BOLD}║  ✓ ALL SERVICES RUNNING SUCCESSFULLY           ║${NC}"
    echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Monitor both processes (blocking)
    monitor_processes
}

main