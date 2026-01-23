# KBV2 Architecture Specification

## Overview
This document defines the architecture for enhancing KBV2 with Natural Language Query Interface, Domain Tagging, and Human Review UI features as per the implementation plan.

## Core Components

### 1. Natural Language Query Interface
- **text_to_sql_agent.py**: Translates natural language to SQL with validation
- **query_api.py**: FastAPI endpoints for query interface

### 2. Domain Tagging System
- **schema.py modifications**: Add domain columns to entities, edges, documents
- **orchestrator.py modifications**: Propagate domain during document processing
- **SQL migrations**: Add domain columns and indexes

### 3. Human Review Interface
- **review_service.py**: Manages review queue for low-confidence resolutions
- **review_api.py**: API endpoints for review operations
- **schema.py additions**: ReviewQueue table for tracking reviews

### 4. MCP Protocol Layer (Optional)
- **mcp_server.py**: Implements MCP protocol for external tool integration

## Directory Structure
Follows structure defined in implementation plan with new modules under src/knowledge_base/.

## Dependencies
All Python dependencies managed via uv (FastAPI, SQLAlchemy, etc.).

## Verification
Each phase must be verified with `uv run pytest` before proceeding to next phase.