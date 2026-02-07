#!/bin/bash
# KBv2 Quick Start Script
# Usage: ./quick_start.sh [command]
# Commands: verify, start, stop, restart, status, logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           KBv2 Crypto Knowledgebase - Quick Start         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

verify_deployment() {
    echo -e "${BLUE}🔍 Running deployment verification...${NC}"
    uv run python verify_deployment.py
}

start_server() {
    echo -e "${GREEN}🚀 Starting KBv2 server...${NC}"
    
    # Check if systemd service exists
    if [ -f "/etc/systemd/system/kbv2.service" ]; then
        echo "Starting systemd service..."
        sudo systemctl start kbv2
        sleep 2
        sudo systemctl status kbv2 --no-pager
    else
        echo "Starting in foreground (Ctrl+C to stop)..."
        echo "Server will be available at http://localhost:8765"
        echo ""
        PYTHONPATH="src:$PYTHONPATH" uv run python -m knowledge_base.production
    fi
}

stop_server() {
    echo -e "${YELLOW}🛑 Stopping KBv2 server...${NC}"
    
    if [ -f "/etc/systemd/system/kbv2.service" ]; then
        sudo systemctl stop kbv2
        echo -e "${GREEN}✓ Service stopped${NC}"
    else
        echo -e "${YELLOW}⚠ No systemd service found. If running in foreground, press Ctrl+C${NC}"
    fi
}

restart_server() {
    echo -e "${YELLOW}🔄 Restarting KBv2 server...${NC}"
    
    if [ -f "/etc/systemd/system/kbv2.service" ]; then
        sudo systemctl restart kbv2
        sleep 2
        sudo systemctl status kbv2 --no-pager
    else
        echo -e "${RED}No systemd service found. Use start/stop manually.${NC}"
    fi
}

show_status() {
    echo -e "${BLUE}📊 KBv2 Status${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if running
    if curl -s http://localhost:8765/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is running${NC}"
        curl -s http://localhost:8765/api/v2/health | python3 -m json.tool 2>/dev/null || true
    else
        echo -e "${RED}✗ Server is not responding${NC}"
    fi
    
    # Check systemd status if applicable
    if [ -f "/etc/systemd/system/kbv2.service" ]; then
        echo ""
        echo "Systemd Status:"
        sudo systemctl is-active kbv2 --quiet && echo -e "${GREEN}● Active${NC}" || echo -e "${RED}● Inactive${NC}"
    fi
    
    # Check database
    echo ""
    echo "Database:"
    if psql -d knowledge_base -c "SELECT COUNT(*) FROM extraction_experiences" > /dev/null 2>&1; then
        count=$(psql -d knowledge_base -t -c "SELECT COUNT(*) FROM extraction_experiences" 2>/dev/null | xargs)
        echo -e "${GREEN}✓ Connected (${count} experiences)${NC}"
    else
        echo -e "${RED}✗ Not connected${NC}"
    fi
}

show_logs() {
    echo -e "${BLUE}📜 KBv2 Logs${NC}"
    
    if [ -f "/etc/systemd/system/kbv2.service" ]; then
        sudo journalctl -u kbv2 -f
    else
        echo -e "${YELLOW}No systemd logs available. Check console output.${NC}"
    fi
}

install_service() {
    echo -e "${BLUE}⚙️ Installing systemd service...${NC}"
    
    if [ ! -f "scripts/kbv2.service" ]; then
        echo -e "${RED}✗ Service file not found${NC}"
        exit 1
    fi
    
    # Update service file with correct paths
    sed "s|/home/muham/development/kbv2|$(pwd)|g" scripts/kbv2.service > /tmp/kbv2.service
    sed -i "s|User=agentzero|User=$(whoami)|g" /tmp/kbv2.service
    
    sudo cp /tmp/kbv2.service /etc/systemd/system/kbv2.service
    sudo systemctl daemon-reload
    sudo systemctl enable kbv2
    
    echo -e "${GREEN}✓ Service installed. Use './quick_start.sh start' to run.${NC}"
}

show_help() {
    print_header
    echo "Usage: ./quick_start.sh [command]"
    echo ""
    echo "Commands:"
    echo "  verify        Run deployment verification"
    echo "  start         Start the KBv2 server"
    echo "  stop          Stop the KBv2 server"
    echo "  restart       Restart the KBv2 server"
    echo "  status        Show server status"
    echo "  logs          Show server logs"
    echo "  install       Install systemd service"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh verify     # Verify deployment readiness"
    echo "  ./quick_start.sh start      # Start the server"
    echo "  ./quick_start.sh status     # Check server status"
}

# Main
COMMAND=${1:-help}

case $COMMAND in
    verify)
        verify_deployment
        ;;
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    install)
        install_service
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
