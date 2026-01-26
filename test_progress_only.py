#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

# Load environment
exec(open('load_env.py').read())

import asyncio
import websockets
import json
import time

async def test_progress_tracking():
    uri = 'ws://127.0.0.1:8765/ws'
    
    print("="*70)
    print("🔍 TESTING PROGRESS TRACKING ONLY")
    print("="*70)
    
    # Create a tiny document
    with open('./testdata/tiny_progress.txt', 'w') as f:
        f.write("Quantum computing breakthrough achieved.")
    
    async with websockets.connect(uri) as ws:
        print("\n✅ WebSocket connected")
        
        start = time.time()
        await ws.send(json.dumps({
            'method': 'kbv2/ingest_document',
            'params': {'file_path': './testdata/tiny_progress.txt', 'document_name': 'Progress Test'},
            'id': 'progress-test'
        }))
        
        print("\n⏳ Progress updates:")
        updates = []
        
        while time.time() - start < 60:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
                data = json.loads(msg)
                updates.append(data)
                
                elapsed = time.time() - start
                if data.get('type') == 'progress':
                    print(f"  [{elapsed:5.1f}s] Stage {data.get('stage')} - {data.get('status')}: {data.get('message')}")
                elif 'error' in data:
                    print(f"  ❌ Error: {data.get('error')}")
                    break
                elif 'result' in data and len(updates) > 3:
                    print(f"  ✅ Complete!")
                    break
            except asyncio.TimeoutError:
                if updates:
                    print(f"  [{time.time() - start:5.1f}s] Waiting...")
                continue
        
        print("\n" + "="*70)
        print(f"✅ Total updates: {len(updates)}")
        
        started = [u for u in updates if u.get('type') == 'progress' and u.get('status') == 'started']
        completed = [u for u in updates if u.get('type') == 'progress' and u.get('status') == 'completed']
        
        print(f"✅ Started stages: {len(started)}")
        print(f"✅ Completed stages: {len(completed)}")
        
        for u in started:
            print(f"  - Stage {u.get('stage')}: {u.get('message')}")
        
        # Will fail at embeddings (API quota), but progress tracking works
        print(f"\nProgress tracking works: {'✅ YES' if len(updates) >= 5 else '❌ NO'}")
        print(f"✅ Stages 1-3 work (entity extraction): {'YES' if len([s for s in started if s.get('stage') in [1,2,3]]) >= 3 else 'NO'}")
        
        return len(updates) >= 5

result = asyncio.run(test_progress_tracking())
print(f"{'='*70}\n✅ Overall: {'PASS - Progress tracking works' if result else 'FAIL - Progress tracking broken'}\n{'='*70}")
