#!/usr/bin/env python3
import asyncio
import websockets
import json
import time
import sys

async def test_full_pipeline():
    uri = 'ws://127.0.0.1:8765/ws'
    file_path = './testdata/tiny.txt'
    
    # Create small test file
    with open(file_path, 'w') as f:
        f.write("Apple announced iPhone 15. Tim Cook is CEO.")
    
    print("Testing FULL pipeline progress...")
    print(f"Document: {file_path}")
    print("-" * 50)
    
    async with websockets.connect(uri) as ws:
        request = {
            'method': 'kbv2/ingest_document',
            'params': {
                'file_path': file_path,
                'document_name': 'Progress Test'
            },
            'id': 'progress-test'
        }
        
        await ws.send(json.dumps(request))
        
        updates = []
        start = time.time()
        
        while time.time() - start < 60:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1)
                data = json.loads(msg)
                updates.append(data)
                
                elapsed = time.time() - start
                
                if data.get('type') == 'progress':
                    stage = data.get('stage')
                    status = data.get('status')
                    message = data.get('message')
                    print(f'[{elapsed:5.2f}s] Stage {stage} - {status}: {message}')
                elif 'error' in data:
                    print(f'❌ Error: {data.get("error")}')
                    break
                elif 'result' in data and len(updates) > 3:
                    print(f'✅ Success: {data.get("result")}')
                    break
                    
            except asyncio.TimeoutError:
                continue
        
        print("-" * 50)
        print(f"Updates received: {len(updates)}")
        
        # Count stages
        stages = {}
        for u in updates:
            if u.get('type') == 'progress':
                stage = u.get('stage')
                status = u.get('status')
                if stage not in stages:
                    stages[stage] = {'started': False, 'completed': False}
                stages[stage][status] = True
        
        print("Stage completion:")
        for stage in sorted(stages.keys()):
            s = stages[stage]
            if s['started'] and s['completed']:
                print(f"  Stage {stage}: ✅")
            elif s['started']:
                print(f"  Stage {stage}: ⚠️")
            else:
                print(f"  Stage {stage}: ❌")
        
        return len([s for s in stages.values() if s['started'] and s['completed']]) >= 8

result = asyncio.run(test_full_pipeline())
print(f"\n{'✅ PASS' if result else '❌ FAIL'}")
sys.exit(0 if result else 1)
