"""
FastAPI endpoints for the Natural Language Query Interface.
Implements Google Python style guide with type hints and comprehensive docstrings.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .text_to_sql_agent import TextToSQLAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
database_url = os.getenv("DATABASE_URL", "sqlite:///./knowledge_base.db")
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create router
router = APIRouter(
    prefix="/api/v1/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)


def get_text_to_sql_agent() -> TextToSQLAgent:
    """Dependency to get TextToSQLAgent instance.

    Returns:
        TextToSQLAgent: Initialized agent for translating NL to SQL.
    """
    return TextToSQLAgent(engine)


@router.post("/translate", response_model=Dict)
async def translate_query(
    nl_query: str, agent: TextToSQLAgent = Depends(get_text_to_sql_agent)
) -> Dict:
    """Translate natural language query to SQL without executing it.

    Args:
        nl_query: Natural language query string.
        agent: TextToSQLAgent instance (injected by dependency).

    Returns:
        Dictionary containing:
            - sql: The generated SQL statement
            - warnings: List of validation warnings
            - error: Error message (if any)
    """
    try:
        sql, warnings = agent.translate(nl_query)
        return {"sql": sql, "warnings": warnings, "error": None}
    except Exception as e:
        return {"sql": "", "warnings": [], "error": str(e)}


@router.post("/execute", response_model=Dict)
async def execute_query(
    nl_query: str, agent: TextToSQLAgent = Depends(get_text_to_sql_agent)
) -> Dict:
    """Translate and execute natural language query.

    Args:
        nl_query: Natural language query string.
        agent: TextToSQLAgent instance (injected by dependency).

    Returns:
        Dictionary containing:
            - sql: The generated SQL statement
            - results: Query results (if successful)
            - warnings: List of validation warnings
            - error: Error message (if any)
    """
    result = agent.execute_query(nl_query)
    return result


@router.get("/schema", response_model=Dict)
async def get_schema(agent: TextToSQLAgent = Depends(get_text_to_sql_agent)) -> Dict:
    """Get database schema information.

    Args:
        agent: TextToSQLAgent instance (injected by dependency).

    Returns:
        Dictionary containing table names and their columns with data types.
    """
    return agent.schema_cache
