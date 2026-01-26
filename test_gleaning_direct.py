#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import os
os.environ['GOOGLE_API_KEY'] = 'dummy-key'
import asyncio

from knowledge_base.ingestion.v1.gleaning_service import GleaningService
from knowledge_base.common.gateway import GatewayClient

async def test_gleaning():
    print("Testing Gleaning Service directly...")
    print("=" * 50)
    
    # Create gateway and gleaning service
    gateway = GatewayClient()
    gleaning = GleaningService(gateway)
    
    test_text = "Apple Inc. announced iPhone 15. Tim Cook is CEO."
    
    print(f"Test text: {test_text}")
    print("Calling gleaning.extract()...\n")
    
    try:
        result = await gleaning.extract(test_text)
        print(f"✅ SUCCESS!")
        print(f"Entities found: {len(result.entities)}")
        print(f"Edges found: {len(result.edges)}")
        print(f"Info density: {result.information_density}")
        
        for entity in result.entities:
            print(f"  - {entity.name} ({entity.entity_type})")
        
        for edge in result.edges:
            print(f"  - {edge.source} -> {edge.target} ({edge.edge_type})")
        
        return len(result.entities) > 0
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await gateway.close()

result = asyncio.run(test_gleaning())
print(f"\nResult: {'✅ PASS' if result else '❌ FAIL'}")
