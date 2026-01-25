#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Add the project root to sys.path to handle import path issues
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
os.environ["PYTHONPATH"] = str(project_root / "src")

# Import FastAPI to create mock app if needed
from fastapi import FastAPI

# First, try importing the full app
try:
    from knowledge_base.main import app
except ImportError as e:
    print(f"Warning: Could not import full app due to missing dependencies: {e}")
    print("Creating minimal app instance for schema generation...")

    # Create a minimal app with just the routes
    app = FastAPI(
        title="KBV2 Knowledge Base API",
        description="High-fidelity information extraction and graph visualization API",
        version="1.0.0",
    )

    # Import router modules dynamically to avoid dependency errors
    import importlib.util
    import importlib.machinery

    router_files = [
        "knowledge_base/query_api.py",
        "knowledge_base/review_api.py",
        "knowledge_base/graph_api.py",
        "knowledge_base/document_api.py",
    ]

    # Mock dependency-heavy modules to avoid import errors
    sys.modules["knowledge_base.orchestrator"] = type(sys)(
        "knowledge_base.orchestrator"
    )
    sys.modules["knowledge_base.common.aip193_middleware"] = type(sys)(
        "knowledge_base.common.aip193_middleware"
    )
    sys.modules["knowledge_base.common.error_handlers"] = type(sys)(
        "knowledge_base.common.error_handlers"
    )

    for router_file in router_files:
        router_path = project_root / "src" / router_file
        if router_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(
                    router_file.replace("/", ".").replace(".py", ""), router_path
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[router_file.replace("/", ".").replace(".py", "")] = module
                spec.loader.exec_module(module)
                if hasattr(module, "router"):
                    app.include_router(module.router)
            except Exception as import_error:
                print(
                    f"Warning: Could not include router from {router_file}: {import_error}"
                )

    # Add basic routes from main
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "name": "KBV2 Knowledge Base API",
        }

    @app.get("/ready")
    async def readiness_check():
        return {
            "status": "ready",
            "version": "1.0.0",
            "name": "KBV2 Knowledge Base API",
        }


# Generate OpenAPI schema
try:
    openapi_schema = app.openapi()
except Exception as e:
    print(f"Error generating OpenAPI schema: {e}")
    # Fallback to manual schema generation
    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title or "KBV2 Knowledge Base API",
        version=app.version or "1.0.0",
        description=app.description
        or "High-fidelity information extraction and graph visualization API",
        routes=app.routes,
    )

# Ensure the directory exists
output_dir = Path(project_root / "frontend")
output_dir.mkdir(parents=True, exist_ok=True)

# Save the schema to file
output_file = output_dir / "openapi-schema.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

print(f"OpenAPI schema generated successfully at: {output_file}")
print(f"Schema version: {openapi_schema.get('openapi', 'N/A')}")
print(f"API title: {openapi_schema.get('info', {}).get('title', 'N/A')}")
print(f"API version: {openapi_schema.get('info', {}).get('version', 'N/A')}")
print(f"Routes included: {len(openapi_schema.get('paths', {}))}")
