"""Unit tests for the Query API endpoints.

Tests the natural language query interface including SQL translation,
query execution, and schema retrieval functionality.
"""

from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from knowledge_base.main import app

client = TestClient(app)


def test_translate_valid_query():
    """Test translating valid natural language query to SQL."""
    nl_query = "How many documents were uploaded in the last week?"

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.translate.return_value = (
            "SELECT COUNT(*) FROM documents WHERE uploaded_at >= NOW() - INTERVAL '7 days'",
            ["Time range could be ambiguous - 'last week' interpreted as 7 days"]
        )
        mock_agent_class.return_value = mock_agent

        response = client.post(
            "/api/v1/query/translate",
            params={"nl_query": nl_query}
        )

        assert response.status_code == 200
        result = response.json()
        assert "sql" in result
        assert "warnings" in result
        assert "error" in result
        assert result["sql"].startswith("SELECT COUNT(*)")
        assert len(result["warnings"]) == 1
        assert result["error"] is None
        mock_agent.translate.assert_called_once_with(nl_query)


def test_translate_invalid_query():
    """Test translating invalid query returns error."""
    nl_query = "invalid query syntax``"

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.translate.side_effect = Exception("Invalid query syntax")
        mock_agent_class.return_value = mock_agent

        response = client.post(
            "/api/v1/query/translate",
            params={"nl_query": nl_query}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["sql"] == ""
        assert result["warnings"] == []
        assert result["error"] == "Invalid query syntax"
        mock_agent.translate.assert_called_once_with(nl_query)


def test_translate_empty_query():
    """Test translating empty query."""
    nl_query = ""

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.translate.return_value = ("", ["Empty query provided"])
        mock_agent_class.return_value = mock_agent

        response = client.post(
            "/api/v1/query/translate",
            params={"nl_query": nl_query}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["sql"] == ""
        assert len(result["warnings"]) == 1
        assert result["error"] is None


def test_execute_valid_query():
    """Test executing valid natural language query."""
    nl_query = "Show me all documents with 'AI' in the title"

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute_query.return_value = {
            "sql": "SELECT * FROM documents WHERE title ILIKE '%AI%'",
            "results": [
                {"id": 1, "title": "AI in Healthcare", "uploaded_at": "2024-01-15T10:00:00"},
                {"id": 2, "title": "Machine Learning and AI", "uploaded_at": "2024-01-10T14:30:00"}
            ],
            "warnings": [],
            "error": None
        }
        mock_agent_class.return_value = mock_agent

        response = client.post(
            "/api/v1/query/execute",
            params={"nl_query": nl_query}
        )

        assert response.status_code == 200
        result = response.json()
        assert "sql" in result
        assert "results" in result
        assert "warnings" in result
        assert "error" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "AI in Healthcare"
        assert result["error"] is None
        mock_agent.execute_query.assert_called_once_with(nl_query)


def test_execute_query_with_error():
    """Test executing query that generates SQL but fails at runtime."""
    nl_query = "Select from invalid table name"

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute_query.return_value = {
            "sql": "SELECT * FROM nonexistent_table",
            "results": None,
            "warnings": ["Table 'nonexistent_table' may not exist"],
            "error": "relation 'nonexistent_table' does not exist"
        }
        mock_agent_class.return_value = mock_agent

        response = client.post(
            "/api/v1/query/execute",
            params={"nl_query": nl_query}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["sql"] == "SELECT * FROM nonexistent_table"
        assert result["results"] is None
        assert len(result["warnings"]) == 1
        assert result["error"] == "relation 'nonexistent_table' does not exist"


def test_get_schema_success():
    """Test retrieving database schema information."""
    mock_schema = {
        "documents": {
            "id": "UUID",
            "title": "TEXT",
            "content": "TEXT",
            "uploaded_at": "TIMESTAMP",
            "status": "VARCHAR"
        },
        "entities": {
            "id": "UUID",
            "name": "TEXT",
            "type": "VARCHAR",
            "created_at": "TIMESTAMP"
        },
        "edges": {
            "id": "UUID",
            "source_id": "UUID",
            "target_id": "UUID",
            "relationship_type": "VARCHAR",
            "confidence": "FLOAT"
        }
    }

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.schema_cache = mock_schema
        mock_agent_class.return_value = mock_agent

        response = client.get("/api/v1/query/schema")

        assert response.status_code == 200
        result = response.json()
        assert "data" in result
        assert "documents" in result["data"]
        assert "entities" in result["data"]
        assert "edges" in result["data"]
        assert result["data"]["documents"]["title"] == "TEXT"
        assert result["data"]["entities"]["type"] == "VARCHAR"
        assert result["data"]["edges"]["confidence"] == "FLOAT"


def test_get_schema_empty():
    """Test retrieving schema when database is empty."""
    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.schema_cache = {}
        mock_agent_class.return_value = mock_agent

        response = client.get("/api/v1/query/schema")

        assert response.status_code == 200
        result = response.json()
        assert result["data"] == {}


def test_query_with_special_characters():
    """Test query containing special SQL characters."""
    nl_query = "Find documents with citations like 'Smith et al., 2024'"

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.translate.return_value = (
            "SELECT * FROM documents WHERE content LIKE '%Smith et al., 2024%'",
            []
        )
        mock_agent_class.return_value = mock_agent

        response = client.post(
            "/api/v1/query/translate",
            params={"nl_query": nl_query}
        )

        assert response.status_code == 200
        result = response.json()
        assert "Smith et al., 2024" in result["sql"]
        assert result["error"] is None


def test_complex_query_with_joins():
    """Test complex query requiring JOINs between tables."""
    nl_query = "Show me entities related to documents uploaded this month"

    with patch('knowledge_base.query_api.TextToSQLAgent') as mock_agent_class:
        mock_agent = Mock()
        mock_agent.translate.return_value = (
            """SELECT e.name, e.type, d.title
               FROM entities e
               JOIN edges ed ON e.id = ed.source_id
               JOIN documents d ON ed.document_id = d.id
               WHERE d.uploaded_at >= DATE_TRUNC('month', CURRENT_DATE)""",
            ["Query involves multiple table joins"]
        )
        mock_agent_class.return_value = mock_agent

        response = client.post(
            "/api/v1/query/translate",
            params={"nl_query": nl_query}
        )

        assert response.status_code == 200
        result = response.json()
        assert "JOIN" in result["sql"]
        assert "entities" in result["sql"]
        assert len(result["warnings"]) == 1
