#!/usr/bin/env python3
import asyncio
import websockets
import json
import time
import sys

async def test_all_llm_logging():
    uri = 'ws://127.0.0.1:8765/ws'
    file_path = './testdata/simple.txt'
    
    print("=" * 80)
    print("🔍 TESTING WITH COMPREHENSIVE LLM LOGGING")
    print("=" * 80)
    print(f"Backend: {uri}")
    print(f"LLM: http://localhost:8087/v1/")
    print(f"Document: {file_path}")
    print("=" * 80)
    
    # Create simple test file
    with open(file_path, 'w') as f:
        f.write("""
Apple Inc. announced a new product called iPhone 15.
Tim Cook is the CEO of Apple.
The product will be released on September 12, 2024.
""")
    
    print("\n📄 Test document created:")
    with open(file_path) as f:
        print(f.read().strip())
    
    async with websockets.connect(uri, ping_timeout=30) as ws:
        print("\n✅ WebSocket connected")
        
        request = {
            'method': 'kbv2/ingest_document',
            'params': {
                'file_path': file_path,
                'document_name': 'LLM Logging Test'
            },
            'id': 'llm-logging-test'
        }
        
        print("\n📤 Sending request...")
        await ws.send(json.dumps(request))
        
        updates = []
        start_time = time.time()
        
        print("\n⏳ Progress updates:")
        print("-" * 80)
        
        while time.time() - start_time < 180:  # 3 minute max
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                data = json.loads(msg)
                updates.append(data)
                
                elapsed = time.time() - start_time
                
                if data.get('type') == 'progress':
                    stage = data.get('stage', '?')
                    status = data.get('status', '?')
                    message = data.get('message', '')
                    duration = data.get('duration', 0)
                    
                    print(f'[{elapsed:7.2f}s] Stage {str(stage):3s} - {status:>10s}: {message}')
                    
                elif 'error' in data:
                    print(f'\n❌ ERROR: {data.get("error")}')
                    break
                    
                elif 'result' in data and len(updates) > 3:
                    print(f'\n✅ COMPLETE: {json.dumps(data.get("result"))}')
                    break
                    
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if elapsed > 10 and updates:
                    print(f'[{elapsed:7.2f}s] ⏳ Still waiting for LLM response...')
                continue
                
            except Exception as e:
                print(f'[{time.time() - start_time:7.2f}s] ❌ Exception: {e}')
                break
        
        print("-" * 80)
        print(f"\n📊 Summary:")
        print(f"  Total updates: {len(updates)}")
        
        # Show stage completion
        stages = {}
        for u in updates:
            if u.get('type') == 'progress':
                stage = u.get('stage')
                status = u.get('status')
                if stage not in stages:
                    stages[stage] = {'started': False, 'completed': False}
                stages[stage][status] = True
        
        print(f"\n  Stage completion:")
        completed_count = 0
        for stage in sorted(stages.keys()):
            s = stages[stage]
            if s['started'] and s['completed']:
                status = "✅ COMPLETE"
                completed_count += 1
            elif s['started']:
                status = "⚠️  STARTED"
            else:
                status = "❌ NOT STARTED"
            print(f"    Stage {stage}: {status}")
        
        print(f"\n  Completed stages: {completed_count}/{len(stages)}")
        print(f"  Total time: {time.time() - start_time:.1f}s")
        
        # Show backend LLM logs
        print("\n" + "=" * 80)
        print("🔍 BACKEND LLM LOGS:")
        print("=" * 80)
        
        import subprocess
        result = subprocess.run(['tail', '-100', 'backend_clean.log'], capture_output=True, text=True)
        
        llm_logs = []
        for line in result.stdout.split('\n'):
            if any(k in line for k in ['🔍', 'LLM', 'llm', 'PASS', 'gateway', '8087', 'prompt', 'response']):
                if line.strip() and 'kbv2' not in line:
                    llm_logs.append(line)
        
        if llm_logs:
            print("\n".join(llm_logs[-40:]))  # Last 40 lines
        else:
            print("No LLM logs found (backend may not have processed yet)")
        
        return completed_count >= 8

result = asyncio.run(test_all_llm_logging())
print("\n" + "=" * 80)
print(f"✅ PIPELINE SUCCESS: {result}")
print("=" * 80)
sys.exit(0 if result else 1)
