"""Text-to-SQL Agent for translating natural language queries to SQL.
Implements Google Python style guide with type hints and comprehensive docstrings.
"""

import re
from typing import Dict, List, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.sql import quoted_name

try:
    import sqlparse
except ImportError:
    sqlparse = None  # sqlparse is optional


class TextToSQLAgent:
    """Agent that translates natural language queries to SQL statements.

    This class handles the translation of natural language queries into
    executable SQL statements while validating against the database schema.
    """

    def __init__(self, engine: Engine):
        """Initialize the Text-to-SQL agent.

        Args:
            engine: SQLAlchemy engine connected to the target database.
        """
        self.engine = engine
        self.schema_cache = {}
        self._load_schema()

    def _load_schema(self) -> None:
        """Load database schema information for validation."""
        # Get table names
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            )
            tables = [row[0] for row in result]

            # Get column info for each table
            for table in tables:
                result = conn.execute(
                    text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                """),
                    {"table_name": table},
                )
                self.schema_cache[table] = {row[0]: row[1] for row in result}

    def translate(self, nl_query: str) -> Tuple[str, List[str]]:
        """Translate natural language query to SQL.

        Args:
            nl_query: Natural language query string.

        Returns:
            Tuple containing:
                - Generated SQL statement
                - List of validation warnings (empty if no issues)

        Raises:
            ValueError: If query cannot be translated or is invalid.
        """
        # First, check for potential injection patterns in the original query
        security_warnings = self._check_sql_security(nl_query)
        if security_warnings:
            # Return a safe query with security warnings
            return "SELECT NULL LIMIT 0", security_warnings

        # Additional check: if the query looks more like SQL than natural language,
        # flag it as potentially suspicious
        sql_indicators = [
            "1=1",
            " or ",
            " and ",
            "union",
            "drop",
            "delete",
            "update",
            "insert",
            "exec",
            "execute",
            "waitfor",
            "delay",
            "benchmark",
            "sleep",
            "pg_sleep",
        ]
        suspicious_count = 0
        for indicator in sql_indicators:
            if indicator in nl_query.lower():
                suspicious_count += 1
                # Additional check for classic injection patterns
                if indicator == " or " and (
                    "1=1" in nl_query.lower() or "'1'='1'" in nl_query.lower()
                ):
                    security_warnings.append(
                        "Potential SQL injection pattern detected: 'OR 1=1' style attack"
                    )
                    return "SELECT NULL LIMIT 0", security_warnings
                elif (
                    indicator in ["drop", "delete", "update", "insert"]
                    and indicator in nl_query.lower()
                ):
                    security_warnings.append(
                        "Potential SQL injection: DDL/DML command detected in query"
                    )
                    return "SELECT NULL LIMIT 0", security_warnings
                elif "union" in nl_query.lower() and "select" in nl_query.lower():
                    security_warnings.append(
                        "Potential SQL injection: UNION SELECT detected"
                    )
                    return "SELECT NULL LIMIT 0", security_warnings

        # Basic pattern matching for simple queries
        # In a production system, this would use ML/NLP models
        try:
            sql, warnings = self._generate_sql(nl_query)
        except ValueError as e:
            # If generation fails due to suspicious patterns, add security warning
            if suspicious_count > 0:
                security_warnings.append(
                    f"Suspicious pattern detected in query: {str(e)}"
                )
                return "SELECT NULL LIMIT 0", security_warnings
            else:
                raise

        # Validate against schema
        validation_warnings = self._validate_sql(sql)
        warnings.extend(validation_warnings)

        return sql, warnings

    def _generate_sql(self, nl_query: str) -> Tuple[str, List[str]]:
        """Generate SQL from natural language query.

        This is a simplified implementation. In production, this would use
        machine learning models or more sophisticated NLP techniques.

        Args:
            nl_query: Natural language query string.

        Returns:
            Tuple containing generated SQL and any generation warnings.
        """
        warnings = []

        # Convert to lowercase for easier pattern matching
        query_lower = nl_query.lower().strip()

        # Simple pattern: "show me all [table]" -> SELECT * FROM [table]
        if query_lower.startswith("show me all ") or query_lower.startswith(
            "list all "
        ):
            table_name = query_lower.split(" ", 3)[-1].rstrip("?").rstrip(".")
            if table_name in self.schema_cache:
                return f"SELECT * FROM {table_name}", warnings
            else:
                warnings.append(f"Table '{table_name}' not found in schema")
                return f"SELECT * FROM {table_name}", warnings

        # Simple pattern: "find [column] from [table]" -> SELECT [column] FROM [table]
        if query_lower.startswith("find ") and " from " in query_lower:
            parts = query_lower.split(" from ", 1)
            columns_part = parts[0].replace("find ", "")
            table_part = parts[1].rstrip("?").rstrip(".")

            if table_part in self.schema_cache:
                # Split columns by comma and validate
                columns = [col.strip() for col in columns_part.split(",")]
                valid_columns = []
                for col in columns:
                    if col in self.schema_cache[table_part]:
                        valid_columns.append(col)
                    else:
                        warnings.append(
                            f"Column '{col}' not found in table '{table_part}'"
                        )

                if valid_columns:
                    columns_str = ", ".join(valid_columns)
                    return f"SELECT {columns_str} FROM {table_part}", warnings
                else:
                    return f"SELECT * FROM {table_part}", warnings
            else:
                warnings.append(f"Table '{table_part}' not found in schema")
                return f"SELECT * FROM {table_part}", warnings

        # Default case - try to extract table name and do SELECT *
        # Look for common table names in the query
        found_table = None
        for table in self.schema_cache.keys():
            if table in query_lower:
                found_table = table
                break

        if found_table:
            return f"SELECT * FROM {found_table}", warnings

        # If no pattern matches, raise error
        raise ValueError(f"Could not translate query: {nl_query}")

    def _validate_sql(self, sql: str) -> List[str]:
        """Validate generated SQL against database schema and security policies.

        Args:
            sql: Generated SQL statement to validate.

        Returns:
            List of validation warnings.
        """
        warnings = []

        # Security validation first
        security_warnings = self._check_sql_security(sql)
        warnings.extend(security_warnings)

        if security_warnings:  # Don't proceed if there are security issues
            return warnings

        # Schema validation
        # Extract table name from SQL
        table_match = re.search(r"FROM\s+([\w]+)", sql, re.IGNORECASE)
        if not table_match:
            warnings.append("Could not identify table in SQL statement")
            return warnings

        table_name = table_match.group(1)

        # Validate table name for security
        if not self._is_safe_identifier(table_name):
            warnings.append(f"Table name '{table_name}' contains unsafe characters")
            return warnings

        # Check if table exists
        if table_name not in self.schema_cache:
            warnings.append(f"Table '{table_name}' does not exist in schema")
            return warnings

        # Extract selected columns
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE)
        if select_match:
            columns_str = select_match.group(1)
            if columns_str.strip() != "*":
                columns = [col.strip() for col in columns_str.split(",")]
                for col in columns:
                    # Handle aliases (e.g., "column AS alias")
                    if " AS " in col.upper():
                        col = col.split(" AS ")[0].strip()

                    # Validate column name for security
                    if not self._is_safe_identifier(col):
                        warnings.append(
                            f"Column name '{col}' contains unsafe characters"
                        )
                        continue

                    # Check if column exists in table
                    if col not in self.schema_cache[table_name]:
                        warnings.append(
                            f"Column '{col}' does not exist in table '{table_name}'"
                        )

        return warnings

    def _check_sql_security(self, sql: str) -> List[str]:
        """Check SQL for potential security issues.

        Args:
            sql: SQL statement to check.

        Returns:
            List of security warnings.
        """
        warnings = []

        # Check for common SQL injection patterns
        dangerous_patterns = [
            r"(?i)\bDROP\b",  # DROP statements
            r"(?i)\bDELETE\b",  # DELETE statements
            r"(?i)\bUPDATE\b",  # UPDATE statements (unless in SELECT subquery)
            r"(?i)\bINSERT\b",  # INSERT statements
            r"(?i)\bCREATE\b",  # CREATE statements
            r"(?i)\bALTER\b",  # ALTER statements
            r"(?i)\bEXEC\b",  # EXEC statements
            r"(?i)\bEXECUTE\b",  # EXECUTE statements
            r"(?i)\bTRUNCATE\b",  # TRUNCATE statements
            r"(?i)\bMERGE\b",  # MERGE statements
            r"(?i)\bGRANT\b",  # GRANT statements
            r"(?i)\bREVOKE\b",  # REVOKE statements
            r"(?i)\bCOMMIT\b",  # COMMIT statements
            r"(?i)\bROLLBACK\b",  # ROLLBACK statements
            r"(?i)\bSAVEPOINT\b",  # SAVEPOINT statements
            r"(?i)\bUNION\s+ALL\b",  # UNION ALL statements
            r"(?i)UNION(?!\s+SELECT)",  # UNION without SELECT (not in subquery)
            r"(?i)\bWAITFOR\s+DELAY\b",  # SQL Server delay
            r"(?i)';\s*--",  # Comment after statement termination
            r"(?i)';\s*/\*",  # Comment after statement termination
            r"(?i)0x[0-9a-fA-F]+",  # Hexadecimal values (potential bypass)
            r"(?i)char\s*\(",  # Character functions (potential bypass)
            r"(?i)concat\s*\(",  # Concatenation functions (potential bypass)
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, sql):
                # Special case for UPDATE: allow if it's in a subquery context
                if "UPDATE" in pattern and "SELECT" not in sql[: sql.find("UPDATE")]:
                    warnings.append(
                        f"Potentially dangerous SQL pattern detected: {pattern}"
                    )
                # Special case for UNION: allow if it's part of a SELECT statement properly
                elif "UNION" in pattern:
                    # Only warn if UNION is not being used in a legitimate SELECT context
                    if sql.strip().upper().startswith("SELECT"):
                        continue
                    else:
                        warnings.append(
                            f"Potentially dangerous SQL pattern detected: {pattern}"
                        )
                else:
                    warnings.append(
                        f"Potentially dangerous SQL pattern detected: {pattern}"
                    )

        # Check for potentially unsafe operations
        if re.search(r"(?i)(\bOR\b|\bAND\b).*(=|>|<|LIKE)", sql):
            if re.search(r"(?i)(1\s*=\s*1|'[^']+'\s*=\s*'[^']+')", sql):
                warnings.append(
                    "Potential SQL injection condition detected (e.g., '1=1')"
                )

        return warnings

    def _is_safe_identifier(self, identifier: str) -> bool:
        """Check if a SQL identifier is safe to use.

        Args:
            identifier: SQL identifier (table, column, etc.)

        Returns:
            True if the identifier is safe, False otherwise.
        """
        # Only allow alphanumeric characters, underscores, and reasonable length
        return (
            bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier))
            and len(identifier) <= 64
        )

    def execute_query(self, nl_query: str) -> Dict:
        """Translate and execute natural language query.

        Args:
            nl_query: Natural language query string.

        Returns:
            Dictionary containing:
                - sql: The generated SQL statement
                - results: Query results (if successful)
                - warnings: List of validation warnings
                - error: Error message (if any)
        """
        result = {"sql": "", "results": [], "warnings": [], "error": None}

        try:
            sql, warnings = self.translate(nl_query)
            result["sql"] = sql
            result["warnings"] = warnings

            # Final security validation before execution
            if warnings and any("dangerous" in warning.lower() for warning in warnings):
                result["error"] = (
                    "Query contains potentially dangerous patterns and was blocked"
                )
                return result

            # Execute query with additional safety measures
            with self.engine.connect() as conn:
                # Add query timeout to prevent long-running queries
                conn.execute(text("SET statement_timeout = 5000"))  # 5 seconds

                query_result = conn.execute(text(sql))

                # Get column names
                columns = query_result.keys()

                # Convert to list of dictionaries
                rows = []
                row_count = 0
                max_rows = 1000  # Safety limit

                for row in query_result:
                    if row_count >= max_rows:
                        warnings.append(
                            f"Result set truncated at {max_rows} rows for security"
                        )
                        break

                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col] = row[i]
                    rows.append(row_dict)
                    row_count += 1

                result["results"] = rows

        except Exception as e:
            result["error"] = str(e)

        return result
