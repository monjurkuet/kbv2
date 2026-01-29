#!/bin/bash
# KBV2 Server Start Script - Comprehensive Start-up with Logging

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      KBV2 Knowledge Base Server - Smart Start Script       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# === Configuration ===
LOG_DIR="/tmp"
INGESTION_LOG="${LOG_DIR}/kbv2_ingestion.log"
SERVER_LOG="${LOG_DIR}/kbv2_server.log"
SERVER_PORT=8765
STARTUP_WAIT=10

# Display configuration
echo -e "${BLUE}📊 Configuration:${NC}"
echo "   • Log Directory: ${LOG_DIR}"
echo "   • Ingestion Log: ${INGESTION_LOG}"
echo "   • Server Log: ${SERVER_LOG}"
echo "   • Server Port: ${SERVER_PORT}"
echo "   • Startup Wait: ${STARTUP_WAIT}s"
echo ""

# === Pre-flight Checks ===
echo -e "${YELLOW}🔍 Pre-flight Checks:${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ ERROR: 'uv' not found. Please install uv first.${NC}"
    echo "   Installation: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "   ✅ uv package manager found"

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ ERROR: Not in project root directory. Please run from /home/muham/development/kbv2${NC}"
    exit 1
fi
echo "   ✅ Project root directory confirmed"

# Check if log directory exists
if [ ! -d "${LOG_DIR}" ]; then
    echo -e "${RED}❌ ERROR: Log directory ${LOG_DIR} does not exist${NC}"
    exit 1
fi
echo "   ✅ Log directory accessible"
echo ""

# === Stop Existing Instances ===
echo -e "${YELLOW}🛑 Stopping existing KBV2 instances...${NC}"
stopped_count=0

# Find and kill existing processes
if pgrep -f "knowledge-base" > /dev/null; then
    pkill -f "knowledge-base" 2>/dev/null
    stopped_count=$((stopped_count + 1))
    echo "   ✅ Stopped existing knowledge-base processes"
fi

# Wait for clean shutdown
echo "   ⏳ Waiting 3 seconds for clean shutdown..."
sleep 3
echo ""

# === Clear Old Logs ===
echo -e "${YELLOW}🗑️  Log Management:${NC}"

# Backup previous logs if they exist
if [ -f "${INGESTION_LOG}" ]; then
    backup_time=$(date +"%Y%m%d_%H%M%S")
    cp "${INGESTION_LOG}" "${INGESTION_LOG}.${backup_time}.backup"
    echo "   💾 Backed up previous ingestion log to: ${INGESTION_LOG}.${backup_time}.backup"
fi

if [ -f "${SERVER_LOG}" ]; then
    backup_time=$(date +"%Y%m%d_%H%M%S")
    cp "${SERVER_LOG}" "${SERVER_LOG}.${backup_time}.backup"
    echo "   💾 Backed up previous server log to: ${SERVER_LOG}.${backup_time}.backup"
fi

# Start with fresh logs
echo "" > "${INGESTION_LOG}"
echo "" > "${SERVER_LOG}"
echo "   ✅ Fresh log files created"
echo ""

# === Start Server ===
echo -e "${GREEN}🚀 Starting KBV2 Server...${NC}"
echo "   📦 Command: uv run knowledge-base"
echo ""

# Start server in background and capture PID
nohup uv run knowledge-base > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo -e "${GREEN}✅ Server started with PID: ${SERVER_PID}${NC}"
echo ""

# === Wait for Initialization ===
echo -e "${YELLOW}⏱️  Waiting ${STARTUP_WAIT}s for server initialization...${NC}"

# Show progress dots
for i in $(seq 1 ${STARTUP_WAIT}); do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# === Verify Server Status ===
echo -e "${YELLOW}🔍 Verifying server status...${NC}"

if ps -p ${SERVER_PID} > /dev/null; then
    echo -e "${GREEN}✅ Server is running!${NC}"
    echo ""

    # Display server information
    echo -e "${BLUE}📍 Server Information:${NC}"
    echo "   🌐 Server URL: http://localhost:${SERVER_PORT}"
    echo "   🔢 Process ID: ${SERVER_PID}"
    echo ""

    echo -e "${BLUE}📊 Log & Monitor Commands:${NC}"
    echo "   📋 View ingestion logs:  ${YELLOW}tail -f ${INGESTION_LOG}${NC}"
    echo "   💻 View server logs:     ${YELLOW}tail -f ${SERVER_LOG}${NC}"
    echo "   📈 View live logs:       ${YELLOW}multitail ${INGESTION_LOG} ${SERVER_LOG}${NC}"
    echo ""

    echo -e "${BLUE}📝 Testing Commands:${NC}"
    echo "   🧪 Run ingestion test:   ${YELLOW}nohup uv run python -m knowledge_base.clients.cli ingest /tmp/comprehensive_test_data.md > ingestion_test.log 2>&1 &${NC}"
    echo ""

    echo -e "${BLUE}🛠️  Management Commands:${NC}"
    echo "   🛑 Stop server:          ${YELLOW}pkill -f knowledge-base${NC}"
    echo "   📊 Check server status:  ${YELLOW}ps aux | grep knowledge-base${NC}"
    echo "   🔍 Check logs:           ${YELLOW}cat ${LOG_DIR}/kbv2_*.log${NC}"
    echo ""

    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ KBV2 Server Successfully Started                     ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"

else
    echo -e "${RED}❌ Failed to start server${NC}"
    echo ""
    echo -e "${YELLOW}🔍 Troubleshooting:${NC}"
    echo "   📄 Check server logs for errors:"
    echo "      ${YELLOW}tail -50 ${SERVER_LOG}${NC}"
    echo ""
    echo "   🐛 Common issues:"
    echo "      1. Port ${SERVER_PORT} already in use"
    echo "      2. Database connection issues"
    echo "      3. Missing environment variables"
    echo "      4. Dependency problems"
    echo ""
    exit 1
fi
