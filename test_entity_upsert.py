#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import asyncio

from knowledge_base.orchestrator import IngestionOrchestrator

async def test_upsert():
    print("Testing entity upsert functionality...")
    
    orchestrator = IngestionOrchestrator()
    await orchestrator.initialize()
    
    # Create test file with entities that might exist
    with open('./testdata/upsert_test.txt', 'w') as f:
        f.write("Apple announced iPhone 15. Tim Cook is CEO.")
    
    try:
        document = await orchestrator.process_document(
            file_path='./testdata/upsert_test.txt',
            document_name='Upsert Test'
        )
        print(f"✅ Document processed: {document.name}")
        print(f"✅ Status: {document.status}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await orchestrator.close()

result = asyncio.run(test_upsert())
print(f"\nResult: {'✅ PASS' if result else '❌ FAIL'}")
sys.exit(0 if result else 1)
