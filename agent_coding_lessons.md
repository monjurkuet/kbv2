# Agent Coding Lessons

This file documents patterns and lessons learned to prevent repeating mistakes.

## Lesson 001: Environment Variable Loading in Shell Scripts

**Problem:** Shell scripts that check for environment variables fail when those variables are defined in `.env` files but not exported to the shell environment.

**Example:**
```bash
# BAD: Script checks $DATABASE_URL but it's only in .env.production, not exported
if [ -n "$DATABASE_URL" ]; then
    echo "Database URL configured"
fi
# Result: Always fails even when .env.production exists with the variable
```

**Solution:** Source environment files at the start of shell scripts:
```bash
#!/bin/bash
# Load environment variables from .env.production if it exists
if [ -f ".env.production" ]; then
    export $(grep -v '^#' .env.production | xargs)
fi

# Now checks work correctly
if [ -n "$DATABASE_URL" ]; then
    echo "Database URL configured"
fi
```

**When to apply:** Any shell script that validates configuration or connects to external services.

---

## Lesson 002: Database Permission Management

**Problem:** Database migrations run as one user (e.g., `agentzero`) but application/scripts run as another user (e.g., `muham`). Tables are created with owner-only permissions by default.

**Error:**
```
ERROR:  permission denied for table extraction_experiences
```

**Solution:** After migrations, grant appropriate permissions to application users:
```bash
# Grant permissions as the table owner
psql -U <owner> -d <database> -c "GRANT ALL PRIVILEGES ON TABLE <table> TO <app_user>;"
```

**Alternative:** Use PostgreSQL role-based access control with a shared role.

**When to apply:** Multi-user database environments, CI/CD pipelines, development teams.

---

## Lesson 003: Verification Before Marking Complete

**Problem:** Marking tasks complete without verifying they actually work leads to hidden failures.

**Solution:** Always run verification commands after fixes:
```bash
# After applying fix, run the check again
./deployment_checklist.sh
```

**When to apply:** Always. Every fix must be verified.

---

## Lesson 004: Check Both File Existence AND Functionality

**Problem:** Scripts may check that files exist but not verify they actually work (e.g., database migrations exist but tables aren't accessible).

**Solution:** Verify end-to-end functionality:
```bash
# Check file exists
[ -f "migration.py" ] && echo "Migration file exists"

# ALSO verify the result of the migration works
psql -d database -c "SELECT COUNT(*) FROM table" > /dev/null 2>&1
```

**When to apply:** Infrastructure checks, deployment validation, health checks.

---

## Lesson 005: Python Environment Loading in Verification Scripts

**Problem:** Python verification scripts that check for environment variables fail when those variables are only defined in `.env` files.

**Example:**
```python
# BAD: Script checks os.getenv('DATABASE_URL') but it's only in .env
import os
db_url = os.getenv('DATABASE_URL')  # Returns None
```

**Solution:** Load environment files at the start of Python scripts:
```python
import os

# Load environment variables from .env.production if it exists
env_file = os.path.join(os.path.dirname(__file__), ".env.production")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)

# Now checks work correctly
db_url = os.getenv('DATABASE_URL')  # Returns value from .env.production
```

**Alternative:** Use `python-dotenv` library:
```python
from dotenv import load_dotenv
load_dotenv('.env.production')
```

**When to apply:** Deployment verification scripts, test scripts, standalone utilities.

---

## Lesson 006: Provide Multiple Deployment Interfaces

**Problem:** Different deployment scenarios need different interfaces (developer vs operator, foreground vs background, testing vs production).

**Solution:** Provide multiple entry points:
```bash
# For developers - quick verification
./quick_start.sh verify

# For operators - systemd service
./quick_start.sh install
sudo systemctl start kbv2

# For testing - direct Python
uv run python verify_deployment.py

# For production - standalone module
uv run python -m knowledge_base.production
```

**Key Principle:** Meet users where they are - don't force a single workflow.

**When to apply:** Any system with multiple user types or deployment scenarios.

---

## Lesson 007: Create Deployment Status Documents

**Problem:** After complex implementations, it's hard to remember what's been done and what's ready.

**Solution:** Create a `DEPLOY_STATUS.md` that:
- Lists all components with their status
- Shows verification results
- Provides quick reference for commands
- Documents known limitations
- Serves as a go/no-go checklist

**When to apply:** Any multi-component deployment, production readiness reviews, handoffs to operations teams.

---

## Lesson 008: Python Module Import with uv run

**Problem:** When using `uv run python -m module.name`, Python can't find the module even though the file exists.

**Example:**
```bash
# This fails because PYTHONPATH doesn't include 'src'
uv run python -m knowledge_base.production
# Error: No module named knowledge_base.production
```

**Solution:** Set PYTHONPATH to include the src directory:
```bash
# Correct way
PYTHONPATH="src:$PYTHONPATH" uv run python -m knowledge_base.production
```

**In Scripts:**
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHONPATH="src:$PYTHONPATH" uv run python -m knowledge_base.production
```

**Alternative:** Create a wrapper script that exports PYTHONPATH:
```bash
export PYTHONPATH="/path/to/project/src:$PYTHONPATH"
uv run python -m module.name
```

**When to apply:** Any project using `uv run` with module imports from a `src` directory structure.

---

## Lesson 009: Uvicorn Module Loading

**Problem:** When running uvicorn programmatically with `python -m`, the module path string may not resolve correctly.

**Example:**
```python
# This fails when run with `python -m knowledge_base.production`
uvicorn.run("production:app", ...)  # Can't find 'production'
uvicorn.run("knowledge_base.production:app", ...)  # Can't find 'knowledge_base.production'
```

**Solution:** Pass the app object directly instead of a string:
```python
# Correct - pass the actual app object
uvicorn.run(app, host=host, port=port, ...)
```

**Trade-off:** Can't use reload=True with direct object, but for production that's correct anyway.

**When to apply:** Any FastAPI/ASGI app run programmatically with uvicorn.
