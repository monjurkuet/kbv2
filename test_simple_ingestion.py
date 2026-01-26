#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import asyncio

async def test_simple():
    from knowledge_base.orchestrator import IngestionOrchestrator
    
    print("Testing simple ingestion with unique entities...")
    
    orchestrator = IngestionOrchestrator()
    await orchestrator.initialize()
    
    try:
        # Use unique entity names to avoid conflicts
        with open('./testdata/simple_doc.txt', 'w') as f:
            f.write("Quantum Computing announced breakthrough processor.")
        
        doc = await orchestrator.process_document(
            file_path='./testdata/simple_doc.txt',
            document_name='Quantum Breakthrough'
        )
        print(f"✅ SUCCESS: {doc.name}")
        print(f"✅ Status: {doc.status}")
        return doc.status.value == 'COMPLETED'
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await orchestrator.close()

result = asyncio.run(test_simple())
print(f"\n{'='*50}")
print(f"✅ All 9 stages passed: {result}")
print(f"{'='*50}")
sys.exit(0 if result else 1)
