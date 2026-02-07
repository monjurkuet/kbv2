#!/usr/bin/env python3
"""
KBv2 Production Deployment Verification

This script performs comprehensive checks to verify the system is production-ready.
Run this after deployment to ensure everything is working correctly.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import List, Tuple

# Load environment variables from .env.production if it exists
env_file = os.path.join(os.path.dirname(__file__), ".env.production")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class DeploymentVerifier:
    """Comprehensive deployment verification."""

    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []
        self.warnings: List[str] = []

    def check(self, name: str, passed: bool, message: str = ""):
        """Record a check result."""
        self.checks.append((name, passed, message))
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if message and not passed:
            print(f"     └─ {message}")
        return passed

    def warn(self, message: str):
        """Record a warning."""
        self.warnings.append(message)
        print(f"  ⚠️  {message}")

    def print_summary(self):
        """Print final summary."""
        passed = sum(1 for _, p, _ in self.checks if p)
        total = len(self.checks)

        print("\n" + "=" * 60)
        print("DEPLOYMENT VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Checks Passed: {passed}/{total}")
        print(f"Warnings: {len(self.warnings)}")

        if passed == total:
            print("\n🎉 ALL CHECKS PASSED - SYSTEM IS PRODUCTION READY!")
            return 0
        else:
            failed = [name for name, p, _ in self.checks if not p]
            print(f"\n❌ FAILED CHECKS:")
            for name in failed:
                print(f"   - {name}")
            print("\n⚠️  Please fix the failed checks before deploying to production.")
            return 1


async def run_verification():
    """Run all deployment verification checks."""
    verifier = DeploymentVerifier()

    print("\n🔍 KBv2 Production Deployment Verification")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # 1. Environment Checks
    print("1️⃣  Environment Configuration")
    print("-" * 40)

    required_env_vars = [
        "DATABASE_URL",
        "LLM_API_BASE",
        "EMBEDDING_API_BASE",
    ]

    for var in required_env_vars:
        value = os.getenv(var)
        verifier.check(
            f"Environment variable: {var}", bool(value), f"Not set" if not value else ""
        )

    # 2. Module Import Checks
    print("\n2️⃣  Module Imports")
    print("-" * 40)

    modules_to_test = [
        (
            "ExperienceBank",
            "knowledge_base.intelligence.v1.self_improvement.experience_bank",
        ),
        (
            "PromptEvolutionEngine",
            "knowledge_base.intelligence.v1.self_improvement.prompt_evolution",
        ),
        (
            "OntologyValidator",
            "knowledge_base.intelligence.v1.self_improvement.ontology_validator",
        ),
        ("SelfImprovingOrchestrator", "knowledge_base.orchestrator_self_improving"),
        ("ProductionConfig", "knowledge_base.config.production"),
        ("MetricsCollector", "knowledge_base.monitoring.metrics"),
        ("KBv2DataConnector", "knowledge_base.data_pipeline.connector"),
    ]

    for class_name, module_path in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            verifier.check(f"Import: {class_name}", True)
        except Exception as e:
            verifier.check(f"Import: {class_name}", False, str(e))

    # 3. Database Checks
    print("\n3️⃣  Database Connectivity")
    print("-" * 40)

    try:
        import psycopg2

        database_url = os.getenv("DATABASE_URL", "")
        # Parse connection string
        if database_url:
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()

            # Check extraction_experiences table
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'extraction_experiences'
                );
            """)
            table_exists = cursor.fetchone()[0]
            verifier.check("Table: extraction_experiences", table_exists)

            # Check required columns
            if table_exists:
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'extraction_experiences';
                """)
                columns = [row[0] for row in cursor.fetchall()]
                required_columns = [
                    "id",
                    "text_snippet",
                    "entities",
                    "relationships",
                    "domain",
                    "quality_score",
                    "created_at",
                ]
                for col in required_columns:
                    verifier.check(f"Column: {col}", col in columns)

            conn.close()
        else:
            verifier.check("Database connection", False, "DATABASE_URL not set")
    except Exception as e:
        verifier.check("Database connectivity", False, str(e))

    # 4. External Services Checks
    print("\n4️⃣  External Services")
    print("-" * 40)

    import urllib.request

    # Check LLM API
    llm_base = os.getenv("LLM_API_BASE", "")
    if llm_base:
        try:
            req = urllib.request.Request(
                f"{llm_base}/models",
                headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                verifier.check("LLM API", response.status == 200)
        except Exception as e:
            verifier.check("LLM API", False, str(e))
    else:
        verifier.check("LLM API", False, "LLM_API_BASE not set")

    # Check Embedding API
    embedding_base = os.getenv("EMBEDDING_API_BASE", "")
    if embedding_base:
        try:
            req = urllib.request.Request(f"{embedding_base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                verifier.check("Embedding API", response.status == 200)
        except Exception as e:
            verifier.check("Embedding API", False, str(e))
    else:
        verifier.check("Embedding API", False, "EMBEDDING_API_BASE not set")

    # 5. Configuration Checks
    print("\n5️⃣  Production Configuration")
    print("-" * 40)

    try:
        from knowledge_base.config.production import get_config, apply_preset

        # Apply production preset
        config = apply_preset("production")
        verifier.check("Production config loads", True)
        verifier.check("Experience Bank enabled", config.enable_experience_bank)
        verifier.check("Prompt Evolution enabled", config.enable_prompt_evolution)
        verifier.check("Ontology Validation enabled", config.enable_ontology_validation)
        verifier.check("Metrics enabled", config.enable_metrics)

        # Check quality thresholds
        verifier.check(
            "Quality threshold >= 0.85",
            config.experience_bank_min_quality >= 0.85,
            f"Current: {config.experience_bank_min_quality}",
        )

    except Exception as e:
        verifier.check("Production configuration", False, str(e))

    # 6. File Structure Checks
    print("\n6️⃣  File Structure")
    print("-" * 40)

    required_files = [
        "src/knowledge_base/intelligence/v1/self_improvement/experience_bank.py",
        "src/knowledge_base/intelligence/v1/self_improvement/prompt_evolution.py",
        "src/knowledge_base/intelligence/v1/self_improvement/ontology_validator.py",
        "src/knowledge_base/orchestrator_self_improving.py",
        "src/knowledge_base/config/production.py",
        "src/knowledge_base/monitoring/metrics.py",
        "src/knowledge_base/data_pipeline/connector.py",
        "src/knowledge_base/production.py",
        "alembic/versions/experience_bank_001.py",
        ".env.production",
        "DEPLOYMENT_GUIDE.md",
        "IMPLEMENTATION_SUMMARY.md",
    ]

    base_dir = os.path.dirname(__file__)
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        verifier.check(f"File: {file_path}", os.path.exists(full_path))

    # 7. Initialization Test (Optional - can be slow)
    print("\n7️⃣  Component Initialization")
    print("-" * 40)

    try:
        from knowledge_base.monitoring.metrics import metrics_collector, health_checker

        verifier.check("Metrics collector created", metrics_collector is not None)
        verifier.check("Health checker created", health_checker is not None)

        # Try health check
        health = await health_checker.check_health()
        verifier.check("Health check executes", health is not None)

    except Exception as e:
        verifier.check("Component initialization", False, str(e))

    # Print final summary
    return verifier.print_summary()


if __name__ == "__main__":
    exit_code = asyncio.run(run_verification())
    sys.exit(exit_code)
