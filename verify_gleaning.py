#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from knowledge_base.ingestion.v1.gleaning_service import GleaningService
from knowledge_base.orchestrator import IngestionOrchestrator

print("Checking Gleaning Service...")
print(f"✓ GleaningService exists: {GleaningService is not None}")
print(f"✓ GleaningService module: {GleaningService.__module__}")

# Check orchestrator
print("\nChecking Orchestrator...")
orchestrator = IngestionOrchestrator()
print(f"✓ Orchestrator created: {orchestrator is not None}")
print(f"✓ Has _gleaning_service: {hasattr(orchestrator, '_gleaning_service')}")
if hasattr(orchestrator, '_gleaning_service'):
    print(f"✓ Gleaning service type: {type(orchestrator._gleaning_service)}")

# Check the method
import inspect
source = inspect.getsource(orchestrator._extract_knowledge)
print(f"\n✓ _extract_knowledge method uses gleaning: {'gleaning_service' in source or 'GleaningService' in source}")
if 'gleaning' in source.lower():
    lines = [line for line in source.split('\n') if 'gleaning' in line.lower()]
    for line in lines:
        print(f"  {line.strip()}")
