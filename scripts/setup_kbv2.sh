#!/bin/bash
# KBV2 Setup Script - Comprehensive system setup for KBV2
# Usage: ./scripts/setup_kbv2.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT=8765
FRONTEND_PORT=3000

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_python() {
    log_info "Checking Python version (3.12 required)..."
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
        
        if [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -ge 12 ]]; then
            log_success "Python 3.12+ found (version $PYTHON_VERSION)"
            return 0
        else
            log_error "Python 3.12+ required, found $PYTHON_VERSION"
        fi
    else
        log_error "Python 3 not found"
    fi
}

check_uv() {
    log_info "Checking uv installation..."
    if command -v uv &>/dev/null; then
        UV_VERSION=$(uv --version 2>&1 | cut -d' ' -f2)
        log_success "uv found (version $UV_VERSION)"
        return 0
    else
        log_warning "uv not found"
        return 1
    fi
}

install_uv() {
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if command -v uv &>/dev/null; then
        log_success "uv installed successfully"
    else
        log_error "uv installation failed"
    fi
}

# Frontend setup functions removed (frontend no longer exists)
# check_bun() {
#     log_info "Checking Bun installation..."
#     if command -v bun &>/dev/null; then
#         BUN_VERSION=$(bun --version)
#         log_success "Bun found (version $BUN_VERSION)"
#         return 0
#     else
#         log_warning "Bun not found"
#         return 1
#     fi
# }
#
# install_bun() {
#     log_info "Installing Bun..."
#     curl -fsSL https://bun.sh/install | bash
#     export BUN_INSTALL="$HOME/.bun"
#     export PATH="$BUN_INSTALL/bin:$PATH"
#     if command -v bun &>/dev/null; then
#         log_success "Bun installed successfully"
#     else
#         log_error "Bun installation failed"
#     fi
# }

check_postgres() {
    log_info "Checking PostgreSQL installation..."
    if command -v psql &>/dev/null; then
        PSQL_VERSION=$(psql --version | head -n1)
        log_success "PostgreSQL client found ($PSQL_VERSION)"
        
        log_info "Testing database connection..."
        if psql "$DATABASE_URL" -c "SELECT 1" &>/dev/null 2>&1; then
            log_success "Database connection successful"
            
            log_info "Checking for pgvector extension..."
            if psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;" &>/dev/null 2>&1; then
                log_success "pgvector extension is available"
            else
                log_warning "pgvector extension may not be installed"
                log_info "To install pgvector:"
                log_info "  Ubuntu/Debian: sudo apt-get install postgresql-16-pgvector"
                log_info "  Docker: make sure your image includes pgvector"
            fi
        else
            log_warning "Database connection failed. Check DATABASE_URL in .env"
        fi
    else
        log_warning "PostgreSQL client not found"
    fi
}

setup_backend() {
    log_info "=== Setting up Backend ==="
    
    cd "$PROJECT_ROOT"
    
    # Install uv if not present
    if ! check_uv; then
        install_uv
    fi
    
    log_info "Installing Python dependencies with uv..."
    uv sync
    log_success "Python dependencies installed"
    
    # Check/create .env
    if [ ! -f .env ]; then
        log_warning ".env file not found"
        if [ -f .env.example ]; then
            log_info "Creating .env from .env.example"
            cp .env.example .env
            log_warning "Please edit .env with your configuration!"
        else
            log_warning "Create .env file with at least DATABASE_URL"
        fi
    fi
    
    # Source .env for subsequent commands
    if [ -f .env ]; then
        source .env
        export DATABASE_URL
    fi
    
    log_info "Generating OpenAPI schema..."
    uv run python scripts/generate_openapi.py
    log_success "OpenAPI schema generated"
    
    log_info "Setting up database..."
    uv run python scripts/setup_db.py
    log_success "Database setup completed"
    
    log_success "✓ Backend setup completed"
}

# Frontend setup removed (frontend no longer exists)
# setup_frontend() {
#     log_info "=== Setting up Frontend ==="
#
#     # Install bun if not present
#     if ! check_bun; then
#         install_bun
#     fi
#
#     cd "$PROJECT_ROOT/frontend"
#
#     log_info "Installing frontend dependencies..."
#     bun install
#     log_success "Frontend dependencies installed"
#
#     log_info "Generating API client..."
#     if [ -f openapi-schema.json ]; then
#         bun run api:generate
#         log_success "API client generated"
#     else
#         log_warning "openapi-schema.json not found"
#         log_info "API client will be generated when backend starts"
#     fi
#
#     log_success "✓ Frontend setup completed"
# }

show_summary() {
    log_info "\n=== Setup Complete ==="
    log_success "All dependencies installed successfully!"
    log_info "\nNext steps:"
    log_info "1. Edit .env file with your configuration if needed"
    log_info "2. Start the backend:"
    log_info "   ./scripts/start_backend.sh # Start backend only"
    log_info "\nDefault URLs:"
    log_info "   Backend API: http://localhost:$BACKEND_PORT"
    log_info "   API Docs:    http://localhost:$BACKEND_PORT/docs"
    log_info "\nNote: Frontend has been removed from this project."
    log_info "      The backend provides all functionality via API endpoints."
}

main() {
    log_info "KBV2 Setup - Checking system requirements..."

    check_python
    check_uv
    # check_bun  # Frontend check removed
    check_postgres

    setup_backend
    # setup_frontend  # Frontend setup removed

    show_summary
}

main
